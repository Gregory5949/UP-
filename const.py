from pathlib import Path
import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOCAL_DATA_PATH = Path('data')
ARTIFACTS_PATH = Path('../Downloads/back2/ML/artifacts1')
DATA_FILE = 'competition_data_final_pqt'
TARGET_FILE = 'public_train.pqt'
LOAD_DATA_PATH = '../Downloads/back2/uploads'
CATEGORICAL_FEATURES = ['cpe_model_name', 'cpe_manufacturer_name', 'part_of_day', 'cpe_type_cd']
SPLIT_SEED = 42