ARTIFACT_DIR="artifacts"

CONFIG_YAML="config/config.yaml"

#Data Ingestion
DB_NAME="house_price_db"
COLLECTION_NAME="house_price"
INGESTION_DIR="data_ingestion"
TRAIN_TEST_SPLIT_RATIO=0.2
FEATURE_STORE_DIR="feature_store"
DATA_FILE="data.csv"
TRAIN_CSVFILE="train.csv"
TEST_CSVFILE="test.csv"

#Data Validation
VALIDATION_STATUS_FILE="validation_status.json"
VALIDATION_DIR="data_validation"
DRIFT_FILE="drift.yaml"

#Data Transformation
TRANSFORMATION_DIR="data_transformation"
TRAIN_DIR="train"
TEST_DIR="test"
TRAIN_NPFILE="train.npy"
TEST_NPFILE="test.npy"
PREPROCESSOR_DIR="preprocessor"
FEATURE_PREPROCESSOR="feature_preprocessor.pkl"
TARGET_PREPROCESSOR="target_preprocessor.pkl"

#Model Trainer
MODEL_DIR="model_trainer"
MODEL_FILE="model.pkl"
EXP_JSON="id.json"

#Model Evaluation
EVALUATION_STATUS_FILE="evaluation_status.json"
EVALUATION_DIR="model_evaluation"

#Model Pusher
MODEL_SAVING_DIR="model/model.pkl"