import os 
from constants import *
from datetime import datetime
from dataclasses import dataclass


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class DataIngestionConfig:
    database_name:str=DB_NAME
    timestamp:str=TIMESTAMP
    collection_name:str=COLLECTION_NAME
    ingestion_dir:str=INGESTION_DIR
    train_test_split_ratio:int=TRAIN_TEST_SPLIT_RATIO
    feature_store_dir=os.path.join(INGESTION_DIR,FEATURE_STORE_DIR)
    train_dir=os.path.join(INGESTION_DIR,TRAIN_DIR)
    test_dir=os.path.join(INGESTION_DIR,TEST_DIR)