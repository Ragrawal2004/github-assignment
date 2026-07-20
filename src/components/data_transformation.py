import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.configuration import Configuration
from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


class DataTransformation:
    """
    Builds the sklearn preprocessing pipeline and applies it to the
    train/test splits produced by DataIngestion.

    Column lists and the target column are read from config.yaml
    (via Configuration) rather than being hardcoded here. Previously
    the same numerical/categorical column lists were duplicated in
    three places (config.yaml, schema.yaml, and this file) -- any one
    of them drifting out of sync would silently break the pipeline.
    This component is now the single point where those columns are
    consumed, not redefined.
    """

    def __init__(self, configuration: Configuration = None):
        configuration = configuration or Configuration()
        self.project_config = configuration.get_config()

    def get_data_transformer(self) -> ColumnTransformer:
        try:
            numerical_columns = self.project_config.features.numerical_columns
            categorical_columns = self.project_config.features.categorical_columns

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_pipeline, numerical_columns),
                    ("cat", categorical_pipeline, categorical_columns),
                ]
            )

            logger.info("Data transformation pipeline created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
        try:
            logger.info("Reading train and test datasets")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer()

            target_column = self.project_config.features.target_column

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logger.info("Applying preprocessing on training data")
            X_train_transformed = preprocessing_obj.fit_transform(X_train)

            logger.info("Applying preprocessing on testing data")
            X_test_transformed = preprocessing_obj.transform(X_test)

            preprocessor_path = self.project_config.artifacts.preprocessor_path
            logger.info("Saving preprocessing object")
            save_object(file_path=preprocessor_path, obj=preprocessing_obj)

            # Persist the transformed arrays too, not just the
            # preprocessor. This makes "data_transformation" a real,
            # independently-cacheable DVC stage: the next stage
            # (model_training) depends on these files on disk rather
            # than on transformation and training being fused into one
            # in-memory Python call.
            save_object(
                file_path=self.project_config.artifacts.transformed_train_path,
                obj=(X_train_transformed, y_train),
            )
            save_object(
                file_path=self.project_config.artifacts.transformed_test_path,
                obj=(X_test_transformed, y_test),
            )

            logger.info("Data Transformation Completed Successfully")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                preprocessor_path,
            )

        except Exception as e:
            raise CustomException(e, sys) from e
