from src.components.dataIngestion import DataIngestion
from src.components.dataTransformation import DataTransformation
from src.components.modelTrainer import ModelTrainer

if __name__ == "__main__":
    ##Data Ingestion
    #Create object for data ingestion
    Object_dataIngestion = DataIngestion()

    #Initiate Data Ingetion
    train_data_path,test_data_path = Object_dataIngestion.initiate_data_ingestion()

    ##Data Transformation
    #Create object for data transformation
    Object_dataTransformation = DataTransformation()

    #Initiate Data Trasformation
    train_arr,test_arr,_ = Object_dataTransformation.initiate_data_transformation(
        train_data_path=train_data_path,
        test_data_path= test_data_path
    )

    ##Model Training
    #Create object for Model training
    Object_modeltraining = ModelTrainer()
    Object_modeltraining.initiate_model_training(
        train_arr=train_arr,
        test_arr=test_arr
    )
    