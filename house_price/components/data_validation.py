import os 
import sys
import json
from logger import logging
from evidently.report import Report
from exception import HousePriceException
from evidently.metric_preset import DataDriftPreset
from utils import load_csv,read_yaml,save_json,save_yaml
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
                    logging.info(f"{column} not present in data extracted")
            return status

        except Exception as e:
            raise HousePriceException(e,sys)

    def detect_data_drift(self,train_data,test_data):

        try:
            data_drift_report = Report(metrics=[DataDriftPreset(),])

            data_drift_report.run(current_data=train_data, reference_data=test_data, column_mapping=None)

            json_report=json.loads(data_drift_report.json())

            save_yaml(json_report,self.data_validation_config.drift_yaml)
            logging.info(f"Drift report saved at {self.data_validation_config.drift_yaml}")

            n_features = json_report['metrics'][0]["result"]["number_of_columns"]
            n_drifted_features = json_report['metrics'][0]["result"]["number_of_drifted_columns"]

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report['metrics'][0]["result"]["dataset_drift"]
            return drift_status

        except Exception as e:
            raise HousePriceException(e,sys)

    def initiate_data_validation(self):
        
        try:
            train_data=load_csv(self.data_validation_config.train_filepath)
            logging.info("Train Data extraction for validation is completed")

            test_data=load_csv(self.data_validation_config.test_filepath)
            logging.info("Test Data extraction for validation is completed")

            train_status=self.validate_columns(train_data)
            logging.info(f"Validation of train data columns completed with Status={train_status}")

            test_status=self.validate_columns(test_data)
            logging.info(f"Validation of test data columns completed with Status={test_status}")

            drift_status=self.detect_data_drift(train_data,test_data)
            logging.info("Data Drift Check completed")

            validation_status=train_status and test_status and not drift_status
            save_json({"status":validation_status},self.data_validation_config.validation_dir)

        except Exception as e:
            raise HousePriceException(e,sys)

if __name__=="__main__":
    try:
        logging.info(">>>>>>>>>> Stage 01 Data Validation initiated <<<<<<<<<<")
        data_validation_config=DataValidationConfig()
        data_validation=DataValidation(data_validation_config)
        data_validation.initiate_data_validation()
        logging.info(">>>>>>>>>> Stage 01 Data Validation completed <<<<<<<<<<")

    except Exception as e:
        raise HousePriceException(e,sys)