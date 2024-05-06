import os
import sys
from dotenv import load_dotenv
from mlflow.client import MlflowClient
from exception import HousePriceException

load_dotenv()

class Mlflow:

    def __init__(self):
        try:
            self.mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
            self.mlflow_client=MlflowClient(tracking_uri=self.mlflow_tracking_uri)
        
        except Exception as e:
            raise HousePriceException(e,sys)


     
