import os
import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
import torch
import bisect
import torch.nn as nn
import pickle
from pathlib import Path
import argparse

np.set_printoptions(suppress=True)
from lib.supervised_dataset import SupervisedDataset


def average(embeddings, weights):
    return (embeddings * weights).sum(1) / weights.sum(1)


class TrainableNet(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, numeric_dim, max_categorical_features):
        super().__init__()

        embedding_dim = embedding_matrix.shape[-1]
        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix), freeze=False)
        self.position_embeddings = nn.Embedding(8192, embedding_dim, max_norm=0.1)

        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(num, hidden_dim, max_norm=0.1) for num in max_categorical_features])
        self.hidden_2 = nn.Linear(4 * embedding_dim, hidden_dim)
        self.attention = nn.Linear(embedding_dim, 1)

        self.hidden_3 = nn.Linear(hidden_dim + numeric_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, 1)
        self.age_projection = nn.Linear(hidden_dim, 4)

    def forward(self, input_ids, counts, unique_counts, numeric_features, categorical_features):
        positions = self.position_embeddings(torch.arange(input_ids.size()[1], dtype=torch.int64)).unsqueeze(0)
        counts = counts.unsqueeze(2)
        unique_counts = unique_counts.unsqueeze(2)
        embedded = self.embeddings(input_ids)

        attention_weights = torch.nn.functional.softmax(
            self.attention(embedded + positions) - 100 * (counts < 0.5).float(), dim=1)

        attended = average(embedded, attention_weights)
        counted = average(embedded, counts)
        log_counted = average(embedded, torch.log1p(counts))
        unique_counted = average(embedded, unique_counts)
        embedded = torch.cat([attended, log_counted, counted, unique_counted], dim=1)
        embedded = torch.nn.functional.relu(self.hidden_2(embedded))

        for idx in range(len(self.categorical_embeddings) - 1):
            embedded = embedded + self.categorical_embeddings[idx](categorical_features[:, idx])

        embedded = torch.cat([embedded, numeric_features], dim=1)
        embedded = torch.nn.functional.relu(self.hidden_3(embedded))

        return self.projection(embedded).squeeze(1), self.age_projection(embedded)


class LightningModel(pl.LightningModule):
    def __init__(
            self,
            base_model,
            lr,
            train_loader,
            val_loader,
            num_updates):
        super().__init__()

        self.base_model = base_model
        self.lr = lr

        self.criterion = nn.BCEWithLogitsLoss()
        self.age_criterion = nn.CrossEntropyLoss()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_updates = num_updates
        self.validation_step_outputs = []

    def forward(self, input_ids, counts, unique_counts, numeric_features, categorical_features):
        return self.base_model.forward(
            input_ids, counts, unique_counts, numeric_features, categorical_features)

    def training_step(self, batch, batch_idx):
        items, weights, unique_weights, numeric_features, categorical_features, labels, age = batch

        logits, age_logits = self.forward(
            items, weights, unique_weights, numeric_features, categorical_features)
        loss = self.criterion(logits, labels) + self.age_criterion(age_logits, age)
        self.log('loss', loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        items, weights, unique_weights, numeric_features, categorical_features, labels, age = batch

        logits, age_logits = self.forward(
            items, weights, unique_weights, numeric_features, categorical_features)

        return {
            'y_true': labels.detach(),
            'y_pred': logits.detach(),
            'age_true': age.detach(),
            'age_pred': np.argmax(age_logits.detach(), axis=-1)
        }

    # def on_validation_epoch_end(self, outputs):
    #     y_true = np.concatenate([np.int32(x['y_true']) for x in outputs])
    #     y_pred = np.concatenate([x['y_pred'] for x in outputs])
    #     # score = roc_auc_score(y_true, y_pred)
    #
    #     age_true = np.concatenate([x['age_true'] for x in outputs])
    #     age_pred = np.concatenate([x['age_pred'] for x in outputs])
    #     age_score = accuracy_score(age_true, age_pred)
    #     age_f1_score = f1_score(age_true, age_pred, average='weighted')
    #
    #     # print(f'auc: {score:.5f}, age accuracy: {age_score:.5f}, age f1: {age_f1_score:.5f}')
    #     print(f'age accuracy: {age_score:.5f}, age f1: {age_f1_score:.5f}')

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 0, self.num_updates)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val_loss',
            'reduce_on_plateau': False,
            'frequency': 1
        }
        return [self.optimizer], [scheduler_config]


class SupervisedTrainer:
    def __init__(
            self,
            embedding_matrix,
            base_model_class,
            lr,
            num_epochs,
            batch_size,
            train_items,
            train_weights,
            train_unique_weights,
            train_numeric_features,
            train_categorical_features,
            train_labels,
            train_age,
            val_items,
            val_weights,
            val_unique_weights,
            val_numeric_features,
            val_categorical_features,
            val_labels,
            val_age):
        embedding_matrix = np.copy(embedding_matrix)
        embedding_matrix[0] = np.mean(embedding_matrix, axis=0)
        embedding_matrix /= np.linalg.norm(embedding_matrix, axis=-1, keepdims=True) + 1e-6

        train_loader = SupervisedDataset(
            train_items,
            train_weights,
            train_unique_weights,
            train_numeric_features,
            train_categorical_features,
            train_labels,
            train_age,
            shuffle=True).loader(batch_size)
        val_loader = SupervisedDataset(
            val_items,
            val_weights,
            val_unique_weights,
            val_numeric_features,
            val_categorical_features,
            val_labels,
            val_age,
            shuffle=False).loader(batch_size)

        num_updates = num_epochs * (len(train_items) + batch_size - 1) // batch_size
        max_categorical_features = [
            max(np.max(train_categorical_features[:, i]), np.max(val_categorical_features[:, i])) + 1
            for i in range(train_categorical_features.shape[-1])
        ]

        base_model = base_model_class(embedding_matrix, 600, train_numeric_features.shape[-1], max_categorical_features)

        self.model = LightningModel(
            base_model,
            lr,
            train_loader,
            val_loader,
            num_updates
        )

        trainer = pl.Trainer(
            max_steps=num_updates,
            num_sanity_val_steps=0,
            accumulate_grad_batches=1,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(self.model)

    def predict(self, items, weights, unique_weights, numeric_features, categorical_features):
        self.model.eval()
        loader = SupervisedDataset(
            items,
            weights,
            unique_weights,
            numeric_features,
            categorical_features,
            np.zeros(len(items), dtype=np.float32),
            np.zeros(len(items), dtype=np.int64),
            shuffle=False
        ).loader(256)

        results = []
        ages = []
        with torch.inference_mode():
            for batch in tqdm(loader):
                result, age = self.model(*[item for item in batch[:5]])
                results.append(torch.sigmoid(result).detach())
                ages.append(torch.nn.functional.softmax(age, dim=1).detach())

        return np.concatenate(results, axis=0), np.concatenate(ages, axis=0)


def age_bucket(x):
    return bisect.bisect_left([17, 25, 55], x)


if __name__ == '__main__':
    LOCAL_DATA_PATH = Path('data')
    ARTIFACTS_PATH = Path('artifacts')
    TARGET_FILE = 'public_train.pqt'
    SPLIT_SEED = 42
    UPLOADS_PATH = Path('uploads')
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedder', dest='embedder', required=True)
    parser.add_argument('--freeze-embeddings', dest='freeze_embeddings', action='store_true')
    parser.add_argument('--output-path', dest='output_path', required=True)
    args = parser.parse_args()

    # data = pq.read_table(
    #     LOCAL_DATA_PATH,
    #     columns=['user_id', 'url_host', 'request_cnt'],
    # )
    '''real_users_features'''
    user_cookies_files = os.listdir(UPLOADS_PATH)

    data = pd.concat([joblib.load(f"{UPLOADS_PATH}/{f}") for f in user_cookies_files])

    with open(args.embedder, 'rb') as f:
        embedder = pickle.load(f)
    data_agg = data.groupby(['user_id', 'url_host']).aggregate(
        [('request_cnt', 'sum'), ('url_host', 'count')]).reset_index()
    user_set = set(data_agg['user_id'])
    user_dict = {user: user_id for user, user_id in zip(user_set, range(len(user_set)))}
    rows = np.array(data_agg['user_id'].map(user_dict))

    cols = np.array(data_agg['url_host'].map(embedder.item_to_id))
    counts = np.array(data_agg['request_cnt'])
    unique_counts = [i[1] for i in counts]
    counts = [i[0] for i in counts]
    df = pd.DataFrame.from_dict({
        'user_id': rows,
        'urls': cols,
        'counts': counts,
        'unique_counts': unique_counts,
    })
    df = df.groupby('user_id').agg({'urls': list, 'counts': list, 'unique_counts': list})
    df = df.reset_index()

    categorical_df = pd.read_csv(ARTIFACTS_PATH / 'real_usr_categorical.csv')
    categorical_df['user_id'] = categorical_df['user_id'].map(user_dict)

    df = df.merge(categorical_df, on='user_id', how='left')
    categorical_features = [f for f in categorical_df.columns if f != 'user_id']
    for f in categorical_features:
        unique_values = list(df[f].unique())
        value2id = {value: idx for idx, value in enumerate(unique_values)}
        df[f] = df[f].apply(lambda model: value2id[model])

    numeric_df = pd.read_csv(ARTIFACTS_PATH / 'real_usr_datetime.csv')
    numeric_df['user_id'] = numeric_df['user_id'].map(user_dict)

    df = df.merge(numeric_df, on='user_id', how='left')
    numeric_features = [f for f in numeric_df.columns if f.startswith('fraction')]

    # targets = pq.read_table(TARGET_FILE).to_pandas()
    # data_agg = data.group_by(['user_id', 'url_host']).aggregate([('request_cnt', 'sum'), ('url_host', 'count')])
    #
    # data_agg = data_agg.to_pandas()
    # user_set = set(data_agg['user_id'])
    # user_dict = {user: user_id for user, user_id in zip(user_set, range(len(user_set)))}
    # rows = np.array(data_agg['user_id'].map(user_dict))
    # cols = np.array(data_agg['url_host'].map(embedder.item_to_id))
    # counts = np.array(data_agg['request_cnt_sum'])
    # unique_counts = np.array(data_agg['url_host_count'])
    # df = pd.DataFrame.from_dict({
    #     'user_id': rows,
    #     'urls': cols,
    #     'counts': counts,
    #     'unique_counts': unique_counts,
    # })
    # df = df.groupby('user_id').agg({'urls': list, 'counts': list, 'unique_counts': list})
    # df = df.reset_index()
    #
    # categorical_df = pd.read_csv(ARTIFACTS_PATH / 'categorical.csv')
    # df = df.merge(categorical_df, on='user_id', how='left')
    # categorical_features = [f for f in categorical_df.columns if f != 'user_id']
    # for f in categorical_features:
    #     unique_values = list(df[f].unique())
    #     value2id = {value: idx for idx, value in enumerate(unique_values)}
    #     df[f] = df[f].apply(lambda model: value2id[model])
    #
    # numeric_df = pd.read_csv(ARTIFACTS_PATH / 'datetime.csv')
    #
    # df = df.merge(numeric_df, on='user_id', how='left')
    # numeric_features = [f for f in numeric_df.columns if f.startswith('fraction')]

    # orig_df = df
    # df = targets.merge(df, on='user_id', how='left')
    # df = df[df['is_male'] != 'NA']
    # df = df.dropna()
    # df['is_male'] = df['is_male'].astype(int)
    # df['age'] = df['age'].apply(age_bucket)
    # train_df, val_df = train_test_split(df, test_size=0.1, random_state=SPLIT_SEED)
    #
    # all_is_male = []
    # all_ages = []
    #
    # orig_is_male = []
    # orig_ages = []
    #
    # base_model_class = TrainableNet
    # lr = 1.5e-3
    # num_epochs = 2
    # batch_size = 16

    # trainer = SupervisedTrainer(
    #     embedder.embeddings,
    #     base_model_class,
    #     lr,
    #     num_epochs,
    #     batch_size,
    #     train_df.urls.values,
    #     train_df.counts.values,
    #     train_df.unique_counts.values,
    #     train_df[numeric_features].values,
    #     train_df[categorical_features].values,
    #     train_df.is_male.values,
    #     train_df.age.values,
    #     val_df.urls.values,
    #     val_df.counts.values,
    #     val_df.unique_counts.values,
    #     val_df[numeric_features].values,
    #     val_df[categorical_features].values,
    #     val_df.is_male.values,
    #     val_df.age.values
    # )
    trainer = joblib.load("trainer.pkl")
    # is_male, age = trainer.predict(
    #     orig_df.urls.values,
    #     orig_df.counts.values,
    #     orig_df.unique_counts.values,
    #     orig_df[numeric_features].values,
    #     orig_df[categorical_features].values
    # )
    # orig_is_male.append(is_male)
    # orig_ages.append(age)

    preds_gen, preds_age = trainer.predict(
        df.urls.values,
        df.counts.values,
        df.unique_counts.values,
        df[numeric_features].values,
        df[categorical_features].values
    )

    preds_gen = (preds_gen >= 0.5).astype(int)
    preds_age = preds_age.argmax(axis=-1)
    print(preds_gen, preds_age)

    # print(roc_auc_score(val_df.is_male, predictions[0]))
    # print(accuracy_score(val_df.age, np.argmax(predictions[1], axis=-1)))
    # all_is_male.append(predictions[0])
    # all_ages.append(predictions[1])

    # orig_df['is_male'] = np.mean(orig_is_male, axis=0)
    # orig_ages = np.mean(orig_ages, axis=0)
    # for i in range(4):
    #     orig_df[f'age_{i}'] = orig_ages[:, i]
    # orig_df = orig_df[[c for c in orig_df.columns if c == 'user_id' or c == 'is_male' or c.startswith('age')]]
    #
    # orig_df.to_csv(args.output_path, index=False)
    #
    # print(roc_auc_score(val_df.is_male, np.mean(all_is_male, axis=0)))
    # print(accuracy_score(val_df.age, np.argmax(np.mean(all_ages, axis=0), axis=-1)))
    # print(f1_score(val_df.age, np.argmax(np.mean(all_ages, axis=0), axis=-1), average='weighted'))
