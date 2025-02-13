# Here we handle categorical and numerical values and generate a pkl file

import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_data_transformer(self):
        try:
            numerical_col = ['Academic_history', 'Attendance', 'Mental_health',
                             'Previous_Scores', 'Academic_support', 'Physical_Activity',]
            categorical_col = ['Parental_support', 'Access_to_resources', 'Extracurricular_Activities',
                               'Motivation_Level', 'Internet_Access', 'Family_Income', 'Understand_ability',
                               'School_Type', 'Peer_Influence', 'Mental_illness', 'Parental_Education_Level',
                               'Distance_from_Home', 'Gender',]
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('scaler', StandardScaler())
            ]
        )

            logging.info("Numerical columns scaled")
            logging.info("Categorical columns encoded & scaled")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_col),
                    ('cat', cat_pipeline, categorical_col)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_path = pd.read_csv(train_path)
            test_path = pd.read_csv(test_path)

            logging.info("Read train & test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer() #should be converted into a pickle file
            target_column = "Exam_Score"
            numerical_col = ['Academic_history', 'Attendance', 'Mental_health',
                             'Previous_Scores', 'Academic_support', 'Physical_Activity',]
            input_feature_train = train_path.drop(target_column, axis=1)
            target_feature_train_df=train_path[target_column]

            input_feature_test_df=test_path.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_path[target_column]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    obj.get_data_transformer()

