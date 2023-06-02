from statistics import mode

import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from flask_dropzone import Dropzone
import pandas as pd
import joblib
from torch import nn
from tqdm import tqdm

from const import *
import pytorch_lightning as pl
from generate_embeddings import MetricTrainer
from lib.embedder import Embedder
from lib.supervised_dataset import SupervisedDataset
from prepare_datetime_features import create_fraction_features
from train_nn import LightningModel, average

app = Flask(__name__)
dropzone = Dropzone(app)


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


def make_embedder(data):
    data_agg = data[['user_id', 'url_host', 'request_cnt']].groupby(['user_id', 'url_host']).aggregate(
        [('request_cnt', 'sum')]).reset_index()
    item_set = set(data_agg['url_host'])
    item_dict = {url: idx for idx, url in enumerate(item_set)}
    user_set = set(data_agg['user_id'])
    user_dict = {user: idx for idx, user in enumerate(user_set)}
    users = np.array(data_agg['user_id'].map(user_dict))
    items = np.array(data_agg['url_host'].map(item_dict))
    counts = np.array(data_agg['request_cnt']).flatten()
    df = pd.DataFrame.from_dict({'user_id': users, 'items': items, 'counts': counts, 'ones': np.ones_like(counts)})
    orig_df = df.groupby('user_id').agg({'items': list, 'counts': list, 'ones': list})
    df = orig_df[orig_df['items'].apply(lambda x: len(x) > 1)]
    trainer = MetricTrainer(len(item_set), 96, 10, 512)
    trainer.fit(df['items'].values, df['ones'].values)
    item_embedder = Embedder(item_dict, trainer.model.embeddings.weight.detach())
    return item_embedder


def make_prediction(data):
    embedder = make_embedder(data)
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

    grouped = data[['user_id', *CATEGORICAL_FEATURES]]
    grouped = grouped.groupby('user_id').agg({f: lambda x: mode(x) for f in CATEGORICAL_FEATURES})
    categorical_df = grouped.reset_index()

    data['date'] = pd.to_datetime(data.date).dt.dayofweek

    df = df.merge(categorical_df, on='user_id', how='left')
    categorical_features = [f for f in categorical_df.columns if f != 'user_id']
    for f in categorical_features:
        unique_values = list(df[f].unique())
        value2id = {value: idx for idx, value in enumerate(unique_values)}
        df[f] = df[f].apply(lambda model: value2id[model])

    daily_features = create_fraction_features(data, 'date')
    part_of_day_features = create_fraction_features(data, 'part_of_day')
    numeric_df = daily_features.merge(part_of_day_features, on='user_id', how='left')

    numeric_df['user_id'] = numeric_df['user_id'].map(user_dict)

    df = df.merge(numeric_df, on='user_id', how='left')
    numeric_features = [f for f in numeric_df.columns if f.startswith('fraction')]

    preds_gen, preds_age = trainer.predict(
        df.urls.values,
        df.counts.values,
        df.unique_counts.values,
        df[numeric_features].values,
        df[categorical_features].values
    )
    preds_gen = (preds_gen >= 0.5).astype(int)
    preds_age = preds_age.argmax(axis=-1)

    return [preds_gen, preds_age]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=['POST', 'GET'])
def upload():
    if request.method == "POST":
        user_cookies_files = os.listdir(UPLOADS_PATH)
        data = pd.concat([joblib.load(f"{UPLOADS_PATH}/{f}") for f in user_cookies_files])

        predictions = make_prediction(data)
        data.date = pd.to_datetime(data.date).dt.dayofweek
        day_of_week = data.groupby('date').agg(cnt=('date', 'count'))
        time_of_day = data.groupby('part_of_day').agg(cnt=('part_of_day', 'count'))
        np.count_nonzero(predictions[1] == 1)
        age_group = [np.count_nonzero(predictions[1] == i) for i in range(1, 4)]
        gender = [np.count_nonzero(predictions[0] == i) for i in range(2)]
        return jsonify(generate_data(age_group,
                                     gender,
                                     [item[0] for item in day_of_week.values.tolist()],
                                     [item[0] for item in time_of_day.values.tolist()]))
    return render_template("index.html")


def generate_data(age_group, gender, day_of_week, time_of_day):
    return {
        'age_group': {
            'data': age_group
        },
        'gender': {
            'data': gender
        },
        'day_of_week': {
            'data': day_of_week
        },
        'time_of_day': {
            'data': time_of_day
        }
    }


if __name__ == "__main__":
    trainer = joblib.load("trainer.pkl")
    app.run()
