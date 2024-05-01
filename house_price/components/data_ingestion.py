import os
import sys
import pandas as pd
from logger import logging
from utils import save_csv
from exception import HousePriceException
from data_access.mongo_db import HousePriceData
from sklearn.model_selection import train_test_split
from entity.config_entity import DataIngestionConfig

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig,houseprice_data: HousePriceData):

        self.data_ingestion_config=data_ingestion_config
        self.house_price_data=houseprice_data

    def split_data(self,data):

        try: 
            train_df,test_df=train_test_split(data,test_size=self.data_ingestion_config.train_test_split_ratio)
            return train_df,test_df

        except Exception as e:
            HousePriceException(e,sys)


    def initiate_data_ingestion(self):

        try:
            self.data=self.house_price_data.extract_data()

            save_csv(self.data,self.data_ingestion_config.feature_store_dir)
            logging.info(f"Data stored in Feature Store at {self.data_ingestion_config.feature_store_dir}")

            train_df,test_df=self.split_data(self.data)
            logging.info("Train test Split completed")

            save_csv(train_df,self.data_ingestion_config.train_filepath)
            logging.info(f"Train Data stored at {self.data_ingestion_config.train_filepath}")
            
            save_csv(test_df,self.data_ingestion_config.test_filepath)
            logging.info(f"Train Data stored at {self.data_ingestion_config.test_filepath}")

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