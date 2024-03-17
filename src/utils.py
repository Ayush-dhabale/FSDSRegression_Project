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
        raise CustomException(e,sys)
