import sys
import pandas as pd
from src.first_ml_project.exception import CustomException
from src.first_ml_project.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path=r'artifacts\model.pkl'
            preprocessor_path=r'artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                hours_studied:int,
                previous_scores:int,
                activities:str,
                sleep_hours:int,
                question_practiced:int):
        self.hours_studied=hours_studied
        self.previous_scores=previous_scores
        self.activities=activities
        self.sleep_hours=sleep_hours
        self.question_practiced=question_practiced

    def get_data_as_data_frame(self):
        try:
            custom_data_input = {
                "Hours Studied": [self.hours_studied],
                "Previous Scores":[self.previous_scores],
                "Extracurricular Activities":[self.activities],
                "Sleep Hours":[self.sleep_hours],
                "Sample Question Papers Practiced":[self.question_practiced]
            }
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomException(e, sys)