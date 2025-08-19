import os
import sys
from src.first_ml_project.exception import CustomException
from src.first_ml_project.logger import logging
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r'notebook\data\Student_Performance.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")

            df["income_cat"] = pd.cut(df["Previous Scores"], bins=[0., 40., 65., 80., 100.], labels=[1,2,3,4])
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in split.split(df, df["income_cat"]):
                strat_train_set = df.iloc[train_index]
                strat_test_set = df.iloc[test_index]
            strat_train_set = strat_train_set.drop("income_cat", axis=1)
            strat_test_set = strat_test_set.drop("income_cat", axis=1)

            strat_train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            strat_test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e)
if __name__ == "__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()