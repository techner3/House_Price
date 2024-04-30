import os
import sys
import pandas as pd
from logger import logging
from utils import save_csv
from exception import HousePriceException
from constants import DATA_FILE
from data_access.mongo_db import HousePriceData
from entity.config_entity import DataIngestionConfig

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig,houseprice_data: HousePriceData):

        self.data_ingestion_config=data_ingestion_config
        self.house_price_data=houseprice_data
    
    def export_data_to_feature_store(self,data):

        try:
            os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_dir),exist_ok=True)
            save_csv(data,self.data_ingestion_config.feature_store_dir)
            logging.info("Data stored in Feature Store")

        except Exception as e:
            raise HousePriceException(e,sys)


    def initiate_data_ingestion(self):

        try:
            self.data=self.house_price_data.extract_data()
            self.export_data_to_feature_store(self.data)

        except Exception as e:
            raise HousePriceException(e,sys)

if __name__=="__main__":

    try:
        logging.info(">>>>>>>>>> Stage 00 Data Ingestion initiated <<<<<<<<<<")
        houseprice_data=HousePriceData()
        data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion(data_ingestion_config,houseprice_data)
        data_ingestion.initiate_data_ingestion()
        logging.info(">>>>>>>>>> Stage 00 Data Ingestion completed <<<<<<<<<<")

    except Exception as e:
            raise HousePriceException(e,sys)