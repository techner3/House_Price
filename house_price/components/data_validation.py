import os 
import sys
from logger import logging
from constants import STATUS_FILE
from exception import HousePriceException
from utils import load_csv,read_yaml,save_json
from entity.config_entity import DataValidationConfig

class DataValidation:

    def __init__(self,data_validation_config:DataValidationConfig):
        self.data_validation_config=data_validation_config
        self.schema=read_yaml(self.data_validation_config.config_yaml)
        self.columns=self.schema["target"]+self.schema["year_features"]+self.schema["numerical_features"]+self.schema["categorical_features"]

    def validate_columns(self,data):

        try:
            status=True
            data_columns=list(data.columns)
            if len(self.columns)!=len(data_columns):
                status=False
                logging.info("Number of Columns doesn't match")
            logging.info("Columns count validation completed")
            for column in data_columns:
                if column not in self.columns:
                    status=False
                    logging.info(f"{column} not present in data extracted from feature store")
            logging.info("Columns validation completed")
            os.makedirs(os.path.dirname(self.data_validation_config.validation_dir),exist_ok=True)
            save_json({"status":status},self.data_validation_config.validation_dir)

        except Exception as e:
            HousePriceException(e,sys)

    def initiate_data_validation(self):
        
        try:
            data=load_csv(self.data_validation_config.data_filepath)
            logging.info("Data extracted from feature store")
            self.validate_columns(data)
        except Exception as e:
            HousePriceException(e,sys)

if __name__=="__main__":
    try:
        logging.info(">>>>>>>>>> Stage 01 Data Validation initiated <<<<<<<<<<")
        data_validation_config=DataValidationConfig()
        data_validation=DataValidation(data_validation_config)
        data_validation.initiate_data_validation()
        logging.info(">>>>>>>>>> Stage 01 Data Validation completed <<<<<<<<<<")

    except Exception as e:
        HousePriceException(e,sys)