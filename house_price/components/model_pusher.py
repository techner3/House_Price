import sys
import mlflow
from logger import logging
from ml.ml_flow import Mlflow
from sklearn.pipeline import Pipeline
from constants import MODEL_SAVING_DIR
from exception import HousePriceException
from entity.config_entity import ModelPusherConfig
from utils import load_json,save_object,load_object

class ModelPusher:

    def __init__(self,model_pusher_config:ModelPusherConfig,ml_flow:Mlflow):
        self.model_pusher_config=model_pusher_config
        self.mlflow_client=ml_flow.mlflow_client
        self.exp_id=load_json(self.model_pusher_config.exp_id_file)["exp_id"]

    def save_locally(self,model):

        try:
            preprocessor=load_object(self.model_pusher_config.feature_preprocessor_path)
            model_with_preprocessor=Pipeline([("preprocessor",preprocessor),("estimator",model)])
            save_object(model_with_preprocessor,MODEL_SAVING_DIR)
        except Exception as e:
            HousePriceException(e,sys)

    def initiate_model_pushing(self):

        try:
            run_id=self.mlflow_client.search_runs(experiment_ids=self.exp_id,order_by=[f"metrics.mae DESC"])[0].info.run_id
            model_info=self.mlflow_client.search_model_versions(filter_string=f"run_id='{run_id}'")[0]
            model=mlflow.sklearn.load_model(model_info.source)
            production_model_info=self.mlflow_client.search_model_versions(filter_string="tags.production='True'")
            
            if len(production_model_info)==0:
                self.mlflow_client.set_model_version_tag(model_info.name, key="production", value="True",version=model_info.version)
            else:
                production_model_info=production_model_info[0]
                self.mlflow_client.set_model_version_tag(production_model_info.name, key="production", value="False",version=production_model_info.version)
                self.mlflow_client.set_model_version_tag(model_info.name, key="production", value="True",version=model_info.version)

            self.save_locally(model)
            

        except Exception as e:
            raise HousePriceException(e,sys)
    
if __name__=="__main__":
     
    try:
        logging.info(">>>>>>>>>> Stage 05 Model Pushing initiated <<<<<<<<<<")
        ml_flow=Mlflow()
        model_pusher_config=ModelPusherConfig()
        model_pusher=ModelPusher(model_pusher_config,ml_flow)
        model_pusher.initiate_model_pushing()
        logging.info(">>>>>>>>>> Stage 05 Model Pushing completed <<<<<<<<<<")

    except Exception as e:
        raise HousePriceException(e,sys)