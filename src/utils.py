import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from src.logger import logging
from src.exception import CustomException


def save_object(file_path,object):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(object,file_obj)

    except Exception as e:
        logging.info("Error while saving the object")
        raise CustomException(e,sys)


def evaluate_model(X_train,X_test,y_train,y_test,models):
    
    report = {}
    try:
        for i in range(len(models)):
            model = list(models.values())[i]

            #Train the model
            model.fit(X_train,y_train)

            #Predict the y value
            y_pred = model.predict(X_test)

            #Calulate the model score
            model_score = r2_score(y_test,y_pred)

            #append the model score in report
            report[list(models.keys())[i]] = model_score * 100

        return report

    except Exception as e:
        logging.info("Error while evaluating the model")
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Error while loading the object")
        raise CustomException(e,sys)



