import os
import sys
import pymongo
import certifi
import numpy as np
import pandas as pd
from logger import logging
from dotenv import load_dotenv
from exception import HousePriceException
from constants import DB_NAME, COLLECTION_NAME

ca = certifi.where()

load_dotenv()

class HousePriceData:

    def __init__(self, db_name:str=DB_NAME, collection_name:str=COLLECTION_NAME):

        self.db_name=db_name
        self.collection_name=collection_name
        self.mongo_db_url=os.getenv("MONGODB_URL")
        if self.mongo_db_url ==None:
            raise Exception(f"Environment key: MONGODB_URL is not set")
        logging.info("Retrieved MongoDB URL")

    def get_mongodb_collection(self):

        try:

            self.mongodb_client=pymongo.MongoClient(self.mongo_db_url, tlsCAFile=ca)
            logging.info("Connected with MongoDB Client")
            self.database_collection = self.mongodb_client[self.db_name][self.collection_name]
            logging.info("Retrieved MongoDB data collection")
            return self.database_collection

        except Exception as e:
            raise HousePriceException(e,sys)

    def extract_data(self,):

        try:
            self.collection=self.get_mongodb_collection()
            df = pd.DataFrame(list(self.collection.find()))
            logging.info("Retrieved data from MongoDB")
            df = df.drop(columns=["_id","Id"], axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise HousePriceException(e,sys)
