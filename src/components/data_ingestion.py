import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.configuration import Configuration
from src.exception import CustomException
from src.logger import logger


class DataIngestion:
    """
    Reads the raw dataset and produces a reproducible train/test split.

    Responsibility: raw data -> train.csv / test.csv on disk.
    Nothing else. Validation, transformation, and modeling are handled
    by other components further down the pipeline.
    """

    def __init__(self, configuration: Configuration = None):
        # Allow a Configuration instance to be injected (useful for
        # testing with a different config.yaml); default to the
        # standard project configuration otherwise.
        configuration = configuration or Configuration()
        self.project_config = configuration.get_config()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        try:
            logger.info("Starting Data Ingestion")

            df = pd.read_csv(self.project_config.data.raw_data_path)
            logger.info(f"Dataset loaded successfully with shape {df.shape}")

            train_data_path = self.project_config.data.train_data_path
            test_data_path = self.project_config.data.test_data_path
            os.makedirs(os.path.dirname(train_data_path), exist_ok=True)

            train_df, test_df = train_test_split(
                df,
                test_size=self.project_config.training.test_size,
                random_state=self.project_config.training.random_state,
            )

            train_df.to_csv(train_data_path, index=False)
            test_df.to_csv(test_data_path, index=False)

            logger.info("Train and Test datasets saved successfully")

            return train_data_path, test_data_path

        except Exception as e:
            logger.error("Error occurred during data ingestion")
            raise CustomException(e, sys) from e
