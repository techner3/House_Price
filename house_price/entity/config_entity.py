import os 
from constants import *
from datetime import datetime
from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
    database_name:str=DB_NAME
    collection_name:str=COLLECTION_NAME
    ingestion_dir:str=INGESTION_DIR
    feature_store_dir=os.path.join(ARTIFACT_DIR,INGESTION_DIR,FEATURE_STORE_DIR)

@dataclass
class DataTransformationConfig:
    train_test_split_ratio:int=TRAIN_TEST_SPLIT_RATIO
    data_filepath=os.path.join(ARTIFACT_DIR,INGESTION_DIR,FEATURE_STORE_DIR,DATA_FILE)
    config_yaml=CONFIG_YAML
    train_file_path=os.path.join(ARTIFACT_DIR,TRANSFORMATION_DIR,TRAIN_DIR)
    test_file_path=os.path.join(ARTIFACT_DIR,TRANSFORMATION_DIR,TEST_DIR)
