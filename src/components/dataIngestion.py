import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.dataTransformation import DataTransformation



#Initializze the Data Ingestion Configuration

@dataclass #decorator
class DataIngestionConfig:
    '''
        Is a Special class, which automatically creates the special methods like __init__ and __repr__ 
        Will use it to store three variables
        1. path to training dataset
        2.path to test dataset
        3. path to row dataset

    '''
    train_dataset_path :str = os.path.join('artifacts','train.csv')
    test_dataset_path :str = os.path.join('artifacts','test.csv')
    raw_dataset_path :str = os.path.join('artifacts','raw.csv')


##Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        #Create a variable which will store the paths to raw,trian,test data
        logging.info("Data Ingestion Configuration Starts")
        self.ingestion_config = DataIngestionConfig()
        logging.info("Data Ingestion Configuration completed")

    
    #Create a function to initiate Data Ingestion
    def initiate_data_ingestion(self):
        logging.info("Data Ingetion process starts")

        try:
            df = pd.read_csv('notebook\data\gemstone.csv')
            logging.info("Read the data as pandas dataframe")

            #make directory to save the raw,train,test data as read from csv file provided
            os.makedirs(os.path.dirname(self.ingestion_config.raw_dataset_path),exist_ok=True)
            

            #Save the raw to respective file
            df.to_csv(self.ingestion_config.raw_dataset_path,index=False)
            logging.info("Saved the raw file")
            logging.info("Spliting the data,train and test")
            
            #split the data,test_size --> ratio of total data considered
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=42)

            #Saving the files
            train_set.to_csv(self.ingestion_config.train_dataset_path,index =False,header= True)
            test_set.to_csv(self.ingestion_config.test_dataset_path,index =False,header= True)
            logging.info("Saved the train and test data as cvs files")

            logging.info("Data Ingestion Process Completed")

            return(
                self.ingestion_config.train_dataset_path,
                self.ingestion_config.test_dataset_path
            )


        except Exception as e:
            logging.info("Error occured druing data Ingestion stage")
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    
    train_arr,test_arr,pickle_file = data_transformation.initiate_data_transformation(
        train_data_path=train_data_path,
        test_data_path=test_data_path
        )