import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model

#Initialize the model configuration

@dataclass
class ModleTrainingConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')

#Create class to train the model
class ModelTrainer:
    def __init__(self):
        logging.info("Starting the Model training confioguration")
        self.model_trainer_config = ModleTrainingConfig()
        logging.info("Model Trainig Configuration completed")

    #Create A function to initiate model training
    def initiate_model_training(self,train_arr,test_arr):
        '''
            Parameters:
            train_arr -> array of training data set
            test_arr -> array of testing data set


            Purpose:
            1.Splits the train,test data as X_train,test and y_train,test
            2.Evaluates the model score
            3.Finds the best model
            4.Saves it as pickle file
        '''
        try:
            #Split the train,test data as X_train,test and y_train,test
            logging.info("Splitting the train,test data as X_train,test and y_train,test")

            X_train,X_test,y_train,y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]

            )

            logging.info("Splitted the data")

            #define the models
            models = {
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet()
    
            }

            #Evaluate the models and create report
            logging.info("Evaluating the model")
            report = evaluate_model(
                X_train= X_train,
                X_test= X_test,
                y_test= y_test,
                y_train= y_train,
                models=models

            )
            logging.info("Evaluated models")
            logging.info(f"Model report{report}")
            print(report)
            print('\n====================================================================================\n')

            #Find the best performing model
            best_model_score = max(sorted(report.values()))

            #Best Model name
            best_model_name = list(report.keys())[
                list(report.values()).index(best_model_score)
            ]

            #Best model
            Best_model = models[best_model_name]
            
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            #Save the best model
            logging.info("Saving the best model as pickle file")

            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                object= Best_model
            )

        except Exception as e:
            logging.info("Error occuered while initiating the model training")
            raise CustomException(e,sys)

            


            
            
