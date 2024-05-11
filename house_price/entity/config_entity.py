import os 
from constants import *
from datetime import datetime
from dataclasses import dataclass

TIMESTAMP=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
@dataclass
class DataIngestionConfig:
    database_name:str=DB_NAME
    collection_name:str=COLLECTION_NAME
    ingestion_dir:str=INGESTION_DIR
    train_test_split_ratio:int=TRAIN_TEST_SPLIT_RATIO
    feature_store_dir=os.path.join(ARTIFACT_DIR,INGESTION_DIR,FEATURE_STORE_DIR,DATA_FILE)
    train_filepath=os.path.join(ARTIFACT_DIR,INGESTION_DIR,TRAIN_DIR,TRAIN_CSVFILE)
    test_filepath=os.path.join(ARTIFACT_DIR,INGESTION_DIR,TEST_DIR,TEST_CSVFILE)

@dataclass
class DataValidationConfig:
    train_filepath=os.path.join(ARTIFACT_DIR,INGESTION_DIR,TRAIN_DIR,TRAIN_CSVFILE)
    test_filepath=os.path.join(ARTIFACT_DIR,INGESTION_DIR,TEST_DIR,TEST_CSVFILE)
    validation_dir=os.path.join(ARTIFACT_DIR,VALIDATION_DIR,VALIDATION_STATUS_FILE)
    drift_yaml=os.path.join(ARTIFACT_DIR,VALIDATION_DIR,DRIFT_FILE)
    config_yaml=CONFIG_YAML

@dataclass
class DataTransformationConfig:
    validation_status=os.path.join(ARTIFACT_DIR,VALIDATION_DIR,VALIDATION_STATUS_FILE)
    train_filepath=os.path.join(ARTIFACT_DIR,INGESTION_DIR,TRAIN_DIR,TRAIN_CSVFILE)
    test_filepath=os.path.join(ARTIFACT_DIR,INGESTION_DIR,TEST_DIR,TEST_CSVFILE)
    config_yaml=CONFIG_YAML
    train_file_path=os.path.join(ARTIFACT_DIR,TRANSFORMATION_DIR,TRAIN_DIR,TRAIN_NPFILE)
    test_file_path=os.path.join(ARTIFACT_DIR,TRANSFORMATION_DIR,TEST_DIR,TEST_NPFILE)
    feature_preprocessor_path=os.path.join(ARTIFACT_DIR,TRANSFORMATION_DIR,PREPROCESSOR_DIR,FEATURE_PREPROCESSOR)
    target_preprocessor_path=os.path.join(ARTIFACT_DIR,TRANSFORMATION_DIR,PREPROCESSOR_DIR,TARGET_PREPROCESSOR)

@dataclass
class ModelTrainerConfig:
    experiment_name: str = TIMESTAMP
    train_file_path=os.path.join(ARTIFACT_DIR,TRANSFORMATION_DIR,TRAIN_DIR,TRAIN_NPFILE)
    test_file_path=os.path.join(ARTIFACT_DIR,TRANSFORMATION_DIR,TEST_DIR,TEST_NPFILE)
    model_dir=os.path.join(ARTIFACT_DIR,MODEL_DIR,MODEL_FILE)
    exp_id_file=os.path.join(ARTIFACT_DIR,MODEL_DIR,EXP_JSON)
    model_params="config/model.yaml"

@dataclass
class RegressionMetrics:
    mse:int
    mae:int
    r2:int

@dataclass
class ModelEvaluationConfig:
    model_dir=os.path.join(ARTIFACT_DIR,MODEL_DIR,MODEL_FILE)
    test_filepath=os.path.join(ARTIFACT_DIR,INGESTION_DIR,TEST_DIR,TEST_CSVFILE)
    model_dir=os.path.join(ARTIFACT_DIR,MODEL_DIR,MODEL_FILE)
    config_yaml=CONFIG_YAML
    feature_preprocessor_path=os.path.join(ARTIFACT_DIR,TRANSFORMATION_DIR,PREPROCESSOR_DIR,FEATURE_PREPROCESSOR)
    target_preprocessor_path=os.path.join(ARTIFACT_DIR,TRANSFORMATION_DIR,PREPROCESSOR_DIR,TARGET_PREPROCESSOR)
    model_evaluation_dir=os.path.join(ARTIFACT_DIR,EVALUATION_DIR,EVALUATION_STATUS_FILE)

@dataclass
class ModelPusherConfig:
    model_dir=os.path.join(ARTIFACT_DIR,MODEL_DIR,MODEL_FILE)
    exp_id_file=os.path.join(ARTIFACT_DIR,MODEL_DIR,EXP_JSON)
    feature_preprocessor_path=os.path.join(ARTIFACT_DIR,TRANSFORMATION_DIR,PREPROCESSOR_DIR,FEATURE_PREPROCESSOR)