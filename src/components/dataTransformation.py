import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler


#Initializze the Data Transformation Configuration

@dataclass
class DataTransformationConfig:
    '''
        Is a Special class, which automatically creates the special methods like __init__ and __repr__ 
        Will use it to store one variables
        path to the preprocessor pickel file
    '''
    preprocessor_obj_file_path :str = os.path.join('artifacts','preprocessor.pkl')

#Create class for data transformation
class DataTransformation:
    def __init__(self):
        logging.info("Data Transformation configuration starts")
        self.transformation_config = DataTransformationConfig()
        logging.info("Data Transformation configuration completed")

    #Create funtion to get object of data transformation
    def get_data_transformation_object(self):

        '''
            Purpose:
            1.Seperates the numerical columns and categorical columns
            2.Defines the custom rankings to the categorical features
            3.Creates pipelines to scale the numerical features and ordinally encode the categorical features
            4.Connectes the two pipelines
            5. return the connected pipeline object
        '''
        try:
            logging.info("Data Transformation starts")

            logging.info("Spliting the features,numerical and categorical")    
            #Define which columns should be ordinally encoded and which should be scaled
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_columns = ['cut', 'color', 'clarity']

            logging.info("Defining the custom ranking of the categorical features")
            #Define the custom ranking of categorical feature
            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Creating piprlines")
            #Create Pipelines
            #Numerical Pipeline

            num_pipeline = Pipeline(
                steps= [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())

                ]
            )

            #Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]   
            )

            logging.info("Created Pipelines")

            #Connect the created pipelines
            logging.info("Connecting the pipelines")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            logging.info("Connected the pipelines")

            return preprocessor
        

        except Exception as e:
            logging.info("Error occured druing data transformation object initialization stage")
            raise CustomException(e,sys)

    #Define a function to initiate the data transformation process    
    def initiate_data_transformation(self,train_data_path,test_data_path):
        '''
            Paramenters:
            train_data_path -> path to the training data file(output of data ingetion process)
            test_data_path ->-> path to the testing data file(output of data ingetion process)
            
            Purpose:
            1.Reads the train and test data files
            2.Obtains preprocessor object through get_data_transformation_object
            3.Splits the train , test data set into input->train, test data and target -> train,test data
            4.Transform the data(input_feature)
            5.Concatenate the transfromed(input_feature) data and target data
            6.Save this data as pickel file
            7.returns this file path ,and concatenated data
        T
        '''
        try:
            logging.info("Data tranformation initiated")
            #Reading the train and test data
            df_train = pd.read_csv(train_data_path)
            df_test = pd.read_csv(test_data_path)

            logging.info("Read the train and test data as pandas Dataframe")
            logging.info(f"Train Dataset head{df_train.head().to_string()}")
            logging.info(f"Test Dataset head{df_test.head().to_string()}")

            logging.info("Obataining the preprocessor object")
            preprocessor_object = self.get_data_transformation_object()

            logging.info("Object obtained")

            #Spliting the train and test data as input and output/target features
            target_feature_name ='price'
            drop_columns = [target_feature_name,'id']
            
            #Input feature
            input_feature_train_df = df_train.drop(columns=drop_columns,axis=1)
            target_feature_train_df = df_train[target_feature_name]

            #Output/target feature
            input_feature_test_df = df_test.drop(columns=drop_columns,axis=1)
            target_feature_test_df = df_test[target_feature_name]

            #Transforming the training ,test data set
            logging.info("Transformimg the dataset")

            input_feature_train_arr = preprocessor_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_object.transform(input_feature_test_df)

            logging.info("Transformed the values")

            
            #Concatenation of the transformed dataset(input feature) and target variables
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            #Saving the transformed data into a pickle file
            logging.info("Saving the dataset as pickle file")
            
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                object= preprocessor_object
            )

            logging.info("Saved the transformed dataset")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error occured during Initiation process")
            raise CustomException(e,sys)






