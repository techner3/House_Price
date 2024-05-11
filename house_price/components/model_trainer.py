import os 
import sys
import json
import mlflow
from logger import logging
from ml.ml_flow import Mlflow
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from exception import HousePriceException
from sklearn.model_selection import GridSearchCV
from entity.config_entity import RegressionMetrics
from sklearn.ensemble import RandomForestRegressor
from entity.config_entity import ModelTrainerConfig
from utils import load_numpy_array_data,read_yaml,load_object,save_object,save_json
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,mlflow:Mlflow):
        self.model_trainer_config=model_trainer_config
        self.mlflow_client=mlflow.mlflow_client
        self.model_params=read_yaml(self.model_trainer_config.model_params)
        self.experiment_id=self.mlflow_client.create_experiment(name=self.model_trainer_config.experiment_name)
        self.models= [('XGBoost', XGBRegressor(), self.model_params["models"]["xgboost"]),
                      ("RF", RandomForestRegressor(), self.model_params["models"]["random_forest"]),
                      ("Ridge", Ridge(), self.model_params["models"]["ridge"])]
    
    def calculate_metrics(self,Y_test,Y_pred):

        try:
            mse=mean_squared_error(Y_test, Y_pred)
            mae=mean_absolute_error(Y_test,Y_pred)
            r2=r2_score(Y_test, Y_pred)
            return RegressionMetrics(mse,mae,r2)

        except Exception as e:
            raise HousePriceException(e,sys)

    def best_model_from_mlflow(self):

        try:
            runs=self.mlflow_client.search_runs(experiment_ids=self.experiment_id,order_by=[f"metrics.mae DESC"])
            best_model = mlflow.sklearn.load_model(self.mlflow_client.search_model_versions(filter_string=f"run_id='{runs[0].info.run_id}'")[0].source)
            return best_model

        except Exception as e:
            raise HousePriceException(e,sys)

    def finding_best_model(self,X_train, Y_train, X_test, Y_test):

        try:
            for name, model, params in self.models:
                with mlflow.start_run(experiment_id=self.experiment_id):

                    mlflow.set_tag("mlflow.runName", name)
                    random = GridSearchCV(estimator=model,param_grid=params,cv=3, n_jobs=-1,scoring='r2')
                    random.fit(X_train, Y_train)

                    best_estimator=random.best_estimator_
                    Y_pred=best_estimator.predict(X_test)
                    metrics=self.calculate_metrics(Y_test,Y_pred)

                    mlflow.log_metrics(metrics.__dict__)
                    
                    mlflow.sklearn.log_model(artifact_path=name,sk_model=best_estimator,registered_model_name=name)

            return self.best_model_from_mlflow()

        except Exception as e:
            raise HousePriceException(e,sys)

    def initiate_model_training(self):

        try:

            train_data=load_numpy_array_data(self.model_trainer_config.train_file_path)
            logging.info("Train data locked and loaded for training")

            test_data=load_numpy_array_data(self.model_trainer_config.test_file_path)
            logging.info("Test data locked and loaded for training")

            X_train, Y_train, X_test, Y_test = (train_data[:, :-1],train_data[:, -1],test_data[:, :-1],test_data[:, -1])

            best_model=self.finding_best_model(X_train, Y_train, X_test, Y_test)
            logging.info("Model experimentation completed and best model found")

            save_object(best_model,self.model_trainer_config.model_dir)
            logging.info(f"Model with preprocessor saved at {self.model_trainer_config.model_dir}")

            save_json({"exp_id":self.experiment_id},self.model_trainer_config.exp_id_file)
            logging.info(f"Saved Exp ID file at {self.model_trainer_config.exp_id_file}")

        except Exception as e:
            raise HousePriceException(e,sys)

if __name__=="__main__":

    try:
        logging.info(">>>>>>>>>> Stage 03 Model Training initiated <<<<<<<<<<")
        model_trainer_config=ModelTrainerConfig()
        ml_flow=Mlflow()
        model_trainer=ModelTrainer(model_trainer_config,ml_flow)
        model_trainer.initiate_model_training()
        logging.info(">>>>>>>>>> Stage 03 Model Training initiated <<<<<<<<<<")

    except Exception as e:
        raise HousePriceException(e,sys)