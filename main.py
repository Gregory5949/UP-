import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_dropzone import Dropzone
import pandas as pd
import joblib
from torch import mode

from const import *
import datetime

from generate_embeddings import MetricTrainer
from lib.embedder import Embedder
from prepare_datetime_features import create_fraction_features

app = Flask(__name__)
dropzone = Dropzone(app)


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
    trainer = MetricTrainer(len(item_set), 96, 30, 512)
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

    # categorical_df = pd.read_csv(ARTIFACTS_PATH / 'real_usr_categorical.csv')
    # categorical_df['user_id'] = categorical_df['user_id'].map(user_dict)

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
    trainer = joblib.load("trainer.pkl")
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=['POST', 'GET'])
def upload():
    if request.method == "POST":
        user_cookies_files = os.listdir(LOAD_DATA_PATH)
        data = pd.concat([joblib.load(f"{LOAD_DATA_PATH}/{f}") for f in user_cookies_files])

        make_prediction(data)
        data.date = pd.to_datetime(data.date).dt.dayofweek
        day_of_week = data.groupby('date').agg(cnt=('date', 'count'))
        time_of_day = data.groupby('part_of_day').agg(cnt=('part_of_day', 'count'))
        age_group = [30, 40, 10]
        gender = [30, 40]
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
    app.run(debug=True)
