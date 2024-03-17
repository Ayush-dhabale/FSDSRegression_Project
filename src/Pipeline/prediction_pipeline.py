import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

#Create a Prediction pipeline class

class PredictionPipeline:
    def __init__(self):
        pass

    def Predict(self,feature):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            logging.info("Loading the preprocessor and model")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            logging.info("Loaded preprocessor and model")

            #Scale the data
            logging.info("Scaling the data")
            scaled_data = preprocessor.transform(feature)

            logging.info("Scaled the data")

            logging.info("Predicting the value")
            #Predict the value
            pred = model.predict(scaled_data)

            return pred

        except Exception as e:
            logging.info("Error at Prediction stage")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(
            self,
            carat : float,
            depth : float,
            table : float,
            x : float,
            y : float,
            z : float,
            cut : str,
            color : str,
            clarity : str):
        
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y 
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity


    def get_data_as_Dataframe(self):
        try:
            custom_data_input = {
                'carat' : [self.carat],
                'depth' : [self.depth],
                'table' : [self.table],
                'x' : [self.x],
                'y' : [self.y],
                'z' : [self.z],
                'cut' : [self.cut],
                'color' : [self.color],
                'clarity' : [self.clarity]
            }

            logging.info("Creating Dataframe")
            df = pd.DataFrame(custom_data_input)
            logging.info("Created dataframe")

            return df

        except Exception as e:
            logging.info("Error while creating the data frame")
            raise CustomData(e,sys)


        

