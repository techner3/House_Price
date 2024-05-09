import sys
import numpy as np
from logger import logging
from sklearn.pipeline import Pipeline
from exception import HousePriceException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from entity.config_entity import DataTransformationConfig
from utils import load_csv,read_yaml,save_numpy_array_data,save_object
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,FunctionTransformer


class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.yr_sold_ix=3
        self.yr_built_ix=0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        new_feature=X[:,self.yr_sold_ix]-X[:,self.yr_built_ix]
        new_df=np.c_[X, new_feature]
        return new_df

class DataTransformation:

    def __init__(self,data_transformation_config:DataTransformationConfig):

        self.data_transformation_config=data_transformation_config
        self.schema=read_yaml(self.data_transformation_config.config_yaml)
        self.numerical_features=[feature for feature in self.schema["numerical_features"] if feature not in self.schema["multi_collinear_columns"] and feature not in self.schema["skewed_columns"]]

    @staticmethod
    def split_data(data,target):

        X=data.drop(target,axis=1)
        Y=data[target]
        return X , Y

    def get_feature_preprocessor_obj(self):

        try:
            year_transformer=Pipeline(steps=[('imputer', KNNImputer()),('featuregenerator', FeatureGenerator()),('year_scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),('encoding', OrdinalEncoder(dtype=int,handle_unknown='use_encoded_value',unknown_value=-1))])
            skewed_transformer=Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('skew_correction', FunctionTransformer(np.log1p)),('skew_scaler', StandardScaler())])
            numeric_transformer=Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
            preprocessor = ColumnTransformer(transformers=[
            ('year_transformer', year_transformer,self.schema["numerical_features"]),
            ('to_drop', "drop", self.schema["multi_collinear_columns"]),
            ('cat_transformer', categorical_transformer, self.schema["categorical_features"]),
            ('skew_transformer', skewed_transformer, self.schema["skewed_columns"]),
            ('num_transformer', numeric_transformer,self.numerical_features),
            ])
            return preprocessor

        except Exception as e:
            raise HousePriceException(e,sys)

    def get_target_preprocessor_obj(self):
        try:
            preprocessor=FunctionTransformer(np.log1p)
            return preprocessor
        except Exception as e:
            raise HousePriceException(e,sys)

    def initiate_data_transformation(self):

        try:

            train_data=load_csv(self.data_transformation_config.train_filepath)
            logging.info("Train Data extraction for transformation completed")

            test_data=load_csv(self.data_transformation_config.test_filepath)
            logging.info("Test Data extraction for transformation completed")

            X_train,Y_train=DataTransformation.split_data(train_data,self.schema["target"])
            logging.info(f"Dependent and Independent variables for train data separated")

            X_test,Y_test=DataTransformation.split_data(test_data,self.schema["target"])
            logging.info(f"Dependent and Independent variables for test data separated")

            feature_preprocessor_obj=self.get_feature_preprocessor_obj()
            logging.info(f"Independent features preprocessor loaded")
            target_preprocessor_obj=self.get_target_preprocessor_obj()
            logging.info(f"Dependent feature preprocessor loaded")

            X_train_transformed=feature_preprocessor_obj.fit_transform(X_train)
            logging.info(f"Pre-processing of Independent Train features completed")

            X_test_transformed=feature_preprocessor_obj.transform(X_test)
            logging.info(f"Pre-processing of Independent Test features completed")

            Y_train_transformed=target_preprocessor_obj.fit_transform(Y_train)
            logging.info(f"Pre-processing of dependent Train feature completed")

            Y_test_transformed=target_preprocessor_obj.transform(Y_test)
            logging.info(f"Pre-processing of dependent Test feature completed")

            save_object(feature_preprocessor_obj,self.data_transformation_config.feature_preprocessor_path)
            logging.info(f"Feature Preprocessor saved at {self.data_transformation_config.feature_preprocessor_path}")
            save_object(target_preprocessor_obj,self.data_transformation_config.target_preprocessor_path)
            logging.info(f"Target Preprocessor saved at {self.data_transformation_config.target_preprocessor_path}")

            train_arr = np.c_[X_train_transformed, Y_train_transformed]
            test_arr = np.c_[X_test_transformed, Y_test_transformed]
            
            save_numpy_array_data(train_arr,self.data_transformation_config.train_file_path)
            logging.info(f"Train data saved in numpy array format at {self.data_transformation_config.train_file_path}")

            save_numpy_array_data(test_arr,self.data_transformation_config.test_file_path)
            logging.info(f"Test data saved in numpy array format at {self.data_transformation_config.test_file_path}")
            

        except Exception as e:
            raise HousePriceException(e,sys)

if __name__=="__main__":

    try:
        logging.info(">>>>>>>>>> Stage 02 Data Transformation initiated <<<<<<<<<<")
        data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation(data_transformation_config)
        data_transformation.initiate_data_transformation()
        logging.info(">>>>>>>>>> Stage 02 Data Transformation completed <<<<<<<<<<")

    except Exception as e:
        raise HousePriceException(e,sys) 