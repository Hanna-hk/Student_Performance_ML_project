import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from src.first_ml_project.exception import CustomException
from src.first_ml_project.logger import logging

from src.first_ml_project.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

hours_studied, previous_scores, sleep_hours = 0, 1, 3
class CombinedAttributesAdder(BaseEstimator, TransformerMixin) :
    def __init__(self, add_rest_to_study = True):
        self.add_rest_to_study = add_rest_to_study
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if hasattr(X, "iloc"):
            X=X.values
        effort = X[:, previous_scores] * X[:, hours_studied]
        if self.add_rest_to_study:
            rest_to_study = X[:, sleep_hours] / X[:, hours_studied]
            return np.c_[X, effort, rest_to_study]
        else:
            return np.c_[X, effort]
        
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        #This function is responsible for data transformation
        try:
            cat_columns = ["Extracurricular Activities"]
            num_columns = [
                "Hours Studied",
                "Previous Scores",
                "Sleep Hours",
                "Sample Question Papers Practiced"
            ]

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])   

            logging.info(f"Categorical columns: {cat_columns}")
            logging.info(f"Numerical columns: {num_columns}")

            full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_columns),
                ("cat", cat_pipeline, cat_columns)
            ])

            return full_pipeline
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = "Performance Index"
            num_columns = [
                "Hours Studied",
                "Previous Scores",
                "Sleep Hours",
                "Sample Question Papers Practiced"
            ]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on train and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)