import sys
import mlflow
from logger import logging
from ml.ml_flow import Mlflow
from exception import HousePriceException
from sklearn.metrics import mean_absolute_error
from utils import load_csv,load_object,read_yaml,save_json
from entity.config_entity import ModelEvaluationConfig
from components.data_transformation import DataTransformation


class ModelEvaluation:

    def __init__(self,model_evaluation_config: ModelEvaluationConfig,ml_flow:Mlflow):
        self.model_evaluation_config=model_evaluation_config
        self.mlflow_client=ml_flow.mlflow_client
        self.schema=read_yaml(self.model_evaluation_config.config_yaml)

    @staticmethod
    def get_production_model(mlflow_client):

        try:
            production_model=mlflow_client.search_model_versions(filter_string="tags.production='True'")
            if len(production_model)==0:
                return None
            else:
                return production_model[0]

        except Exception as e:
            raise HousePriceException(e,sys)

    def initiate_model_evalution(self):

        try:
            production_model_info=ModelEvaluation.get_production_model(self.mlflow_client)
            model_evaluation_status=True

            if production_model_info:

                production_model=mlflow.sklearn.load_model(production_model_info.source)
                logging.info("Production model downloaded")

                data_df=load_csv(self.model_evaluation_config.test_filepath)
                logging.info(f"Test Data downloaded successfully")

                X,Y=DataTransformation.split_data(data_df,self.schema["target"])
                logging.info("Independent and Dependent feature separated")

                new_model=load_object(self.model_evaluation_config.model_dir)
                logging.info(f"Newly trained model loaded successfully")

                target_preprocessor=load_object(self.model_evaluation_config.target_preprocessor_path)
                logging.info(f"Target preprocessor loaded successfully")

                y_transformed=target_preprocessor.transform(Y)
                logging.info(f"Dependent features transformation completed")

                y_pred_production=production_model.predict(X)
                y_pred_newmodel=new_model.predict(X)

                mae_production=mean_absolute_error(y_transformed,y_pred_production)
                mae_newmodel=mean_absolute_error(y_transformed,y_pred_newmodel)

                if mae_newmodel>mae_production:
                    model_evaluation_status=False

            save_json({"status":model_evaluation_status},self.model_evaluation_config.model_evaluation_dir)
            logging.info(f"Evaluation Status stored at {self.model_evaluation_config.model_evaluation_dir}")

        except Exception as e:
            raise HousePriceException(e,sys)

if __name__=="__main__":

    try:
        logging.info(">>>>>>>>>> Stage 04 Model Evaluation initiated <<<<<<<<<<")
        ml_flow=Mlflow()
        model_evaluation_config=ModelEvaluationConfig()
        model_evaluation=ModelEvaluation(model_evaluation_config,ml_flow)
        model_evaluation.initiate_model_evalution()
        logging.info(">>>>>>>>>> Stage 04 Model Evaluation completed <<<<<<<<<<")

    except Exception as e:
        raise HousePriceException(e,sys)