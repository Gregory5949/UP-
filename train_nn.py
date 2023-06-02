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
import argparse
import pyarrow.parquet as pq

from const import LOCAL_DATA_PATH, ARTIFACTS_PATH, TARGET_FILE, SPLIT_SEED
# from main import TrainableNet, SupervisedTrainer

from lib.supervised_dataset import SupervisedDataset

np.set_printoptions(suppress=True)


def average(embeddings, weights):
    return (embeddings * weights).sum(1) / weights.sum(1)


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


def age_bucket(x):
    return bisect.bisect_left([17, 25, 55], x)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--embedder', dest='embedder', required=True)
    parser.add_argument('--freeze-embeddings', dest='freeze_embeddings', action='store_true')
    parser.add_argument('--output-path', dest='output_path', required=True)
    args = parser.parse_args()
    #
    # data = pq.read_table(
    #     LOCAL_DATA_PATH,
    #     columns=['user_id', 'url_host', 'request_cnt'],
    # )
    #
    # with open(args.embedder, 'rb') as f:
    #     embedder = pickle.load(f)
    #
    # data_agg = data.groupby(['user_id', 'url_host']).aggregate(
    #     [('request_cnt', 'sum'), ('url_host', 'count')]).reset_index()
    # user_set = set(data_agg['user_id'])
    # user_dict = {user: user_id for user, user_id in zip(user_set, range(len(user_set)))}
    # rows = np.array(data_agg['user_id'].map(user_dict))
    #
    # cols = np.array(data_agg['url_host'].map(embedder.item_to_id))
    # counts = np.array(data_agg['request_cnt'])
    # unique_counts = [i[1] for i in counts]
    # counts = [i[0] for i in counts]
    # df = pd.DataFrame.from_dict({
    #     'user_id': rows,
    #     'urls': cols,
    #     'counts': counts,
    #     'unique_counts': unique_counts,
    # })
    # df = df.groupby('user_id').agg({'urls': list, 'counts': list, 'unique_counts': list})
    # df = df.reset_index()
    #
    # categorical_df = pd.read_csv(ARTIFACTS_PATH / 'real_usr_categorical.csv')
    # categorical_df['user_id'] = categorical_df['user_id'].map(user_dict)
    #
    # df = df.merge(categorical_df, on='user_id', how='left')
    # categorical_features = [f for f in categorical_df.columns if f != 'user_id']
    # for f in categorical_features:
    #     unique_values = list(df[f].unique())
    #     value2id = {value: idx for idx, value in enumerate(unique_values)}
    #     df[f] = df[f].apply(lambda model: value2id[model])
    #
    # numeric_df = pd.read_csv(ARTIFACTS_PATH / 'real_usr_datetime.csv')
    # numeric_df['user_id'] = numeric_df['user_id'].map(user_dict)
    #
    # df = df.merge(numeric_df, on='user_id', how='left')
    # numeric_features = [f for f in numeric_df.columns if f.startswith('fraction')]
    #
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
    # #
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
    # #
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
    # # trainer = joblib.dump(trainer, "trainer.pkl")
    # is_male, age = trainer.predict(
    #     orig_df.urls.values,
    #     orig_df.counts.values,
    #     orig_df.unique_counts.values,
    #     orig_df[numeric_features].values,
    #     orig_df[categorical_features].values
    # )
    # orig_is_male.append(is_male)
    # orig_ages.append(age)
    # #
    # predictions = trainer.predict(
    #     df.urls.values,
    #     df.counts.values,
    #     df.unique_counts.values,
    #     df[numeric_features].values,
    #     df[categorical_features].values
    # )
    #
    # print(roc_auc_score(val_df.is_male, predictions[0]))
    # print(accuracy_score(val_df.age, np.argmax(predictions[1], axis=-1)))
    # all_is_male.append(predictions[0])
    # all_ages.append(predictions[1])
    #
    # orig_df['is_male'] = np.mean(orig_is_male, axis=0)
    # orig_ages = np.mean(orig_ages, axis=0)
    # # for i in range(4):
    # #     orig_df[f'age_{i}'] = orig_ages[:, i]
    # # orig_df = orig_df[[c for c in orig_df.columns if c == 'user_id' or c == 'is_male' or c.startswith('age')]]
    # #
    # # orig_df.to_csv(args.output_path, index=False)
    # #
    # print(roc_auc_score(val_df.is_male, np.mean(all_is_male, axis=0)))
    # print(accuracy_score(val_df.age, np.argmax(np.mean(all_ages, axis=0), axis=-1)))
    # print(f1_score(val_df.age, np.argmax(np.mean(all_ages, axis=0), axis=-1), average='weighted'))
