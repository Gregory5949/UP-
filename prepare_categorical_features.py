import os

import joblib
import pandas as pd
import pyarrow.parquet as pq
from statistics import mode
from pathlib import Path


if __name__ == '__main__':
    LOCAL_DATA_PATH = Path('data')
    ARTIFACTS_PATH = Path('artifacts')
    UPLOADS_PATH = Path('uploads')
    # DATA_FILE = 'competition_data_final_pqt'
    CATEGORICAL_FEATURES = ['cpe_model_name', 'cpe_manufacturer_name', 'part_of_day', 'cpe_type_cd']

    # data = pq.read_table(
    #     LOCAL_DATA_PATH / DATA_FILE,
    #     columns=['user_id', 'request_cnt'] + CATEGORICAL_FEATURES,
    #     read_dictionary=CATEGORICAL_FEATURES
    # )

    user_cookies_files = os.listdir(UPLOADS_PATH)
    data = pd.concat([joblib.load(f"{UPLOADS_PATH}/{f}") for f in user_cookies_files])

    grouped = data[['user_id', *CATEGORICAL_FEATURES ]]
    grouped = grouped.groupby('user_id').aggregate({f: lambda x: mode(x) for f in CATEGORICAL_FEATURES})
    grouped = grouped.reset_index()
    grouped.to_csv(ARTIFACTS_PATH / 'real_usr_categorical.csv', index=False)
