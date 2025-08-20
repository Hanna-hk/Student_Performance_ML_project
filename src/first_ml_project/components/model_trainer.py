import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import PoissonRegressor, LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score

from src.first_ml_project.exception import CustomException
from src.first_ml_project.logger import logging

from src.first_ml_project.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test =(
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Poisson Regressor": PoissonRegressor(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Poisson Regressor": {
                    'alpha': [0.1, 1.0, 10],
                    'max_iter': [100, 300, 1000]
                },
                "Lasso": {
                    'alpha': [0.1, 1.0, 10],
                    'max_iter': [100, 300, 1000]
                },
                "Ridge": {
                    'alpha': [0.1, 1.0, 10],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, 
                                              y_test=y_test, models=models,params=params)

            ## Best Model Score
            best_model_score = max(sorted(model_report.values()))

            ## Best Model Name 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            ## Threshold
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Found best model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square 
        except Exception as e:
            raise CustomException(e, sys)