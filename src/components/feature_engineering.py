import os
import sys

import pandas as pd

from src.config.configuration import Configuration
from src.exception import CustomException
from src.logger import logger


class FeatureEngineering:
    """
    Extension point for derived/engineered features, sitting between
    DataValidation and DataTransformation in the pipeline.

    The current loan-approval feature set (config/config.yaml) needs no
    derived columns, so `transform` is an explicit identity mapping
    today. It is still a real, wired-in pipeline stage -- reading raw
    train/test CSVs and writing to `data/interim/` -- so that adding a
    derived feature later (e.g. a debt-to-income bucket) means editing
    `transform()` only; no other component or pipeline needs to change.
    """

    def __init__(self, configuration: Configuration = None):
        configuration = configuration or Configuration()
        self.project_config = configuration.get_config()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Identity transform for now -- see class docstring.
        return df

    def apply(self, train_path: str, test_path: str) -> tuple[str, str]:
        try:
            logger.info("Starting feature engineering")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = self.transform(train_df)
            test_df = self.transform(test_df)

            interim_train_path = self.project_config.data.interim_train_path
            interim_test_path = self.project_config.data.interim_test_path

            os.makedirs(os.path.dirname(interim_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(interim_test_path), exist_ok=True)

            train_df.to_csv(interim_train_path, index=False)
            test_df.to_csv(interim_test_path, index=False)

            logger.info("Feature engineering completed successfully")

            return interim_train_path, interim_test_path

        except Exception as e:
            raise CustomException(e, sys) from e
