import os
import sys
import pandas as pd
from logger import logging
from utils import save_csv
from exception import HousePriceException
from constants import TRAIN_FILE,TEST_FILE,ARTIFACT_DIR
from data_access.mongo_db import HousePriceData
from sklearn.model_selection import train_test_split
from entity.config_entity import DataIngestionConfig

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig,houseprice_data: HousePriceData):

        self.data_ingestion_config=data_ingestion_config
        self.house_price_data=houseprice_data
    
    def export_data_to_feature_store(self,data):
        feature_store_dir=os.path.join(ARTIFACT_DIR,self.data_ingestion_config.feature_store_dir,self.data_ingestion_config.timestamp)
        os.makedirs(feature_store_dir,exist_ok=True)
        save_csv(data,os.path.join(feature_store_dir,"data.csv"))
        logging.info("Data stored in Feature Store")


    def split_and_save_data(self,data):
        train_data, test_data = train_test_split(data, test_size=self.data_ingestion_config.train_test_split_ratio)
        logging.info("Train Test Split Done")
        train_dir=os.path.join(ARTIFACT_DIR,self.data_ingestion_config.train_dir,self.data_ingestion_config.timestamp)
        test_dir=os.path.join(ARTIFACT_DIR,self.data_ingestion_config.test_dir,self.data_ingestion_config.timestamp)
        os.makedirs(train_dir,exist_ok=True)
        os.makedirs(test_dir,exist_ok=True)
        save_csv(train_data,os.path.join(train_dir,TRAIN_FILE))
        logging.info(f"Train Data stored in {train_dir}")
        save_csv(test_data,os.path.join(test_dir,TEST_FILE))
        logging.info(f"Test Data stored in {test_dir}")

    def initiate_data_ingestion(self):

        try:
            self.data=self.house_price_data.extract_data()
            self.export_data_to_feature_store(self.data)
            self.split_and_save_data(self.data)

        except Exception as e:
            raise HousePriceException(e,sys)

if __name__=="__main__":
    
    logging.info(">>>>>>>>>> Stage 01 Data Ingestion initiated <<<<<<<<<<")
    houseprice_data=HousePriceData()
    data_ingestion_config=DataIngestionConfig()
    data_ingestion=DataIngestion(data_ingestion_config,houseprice_data)
    data_ingestion.initiate_data_ingestion()
    logging.info(">>>>>>>>>> Stage 01 Data Ingestion completed <<<<<<<<<<")