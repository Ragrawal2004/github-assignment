import os
import sys
import math  # Unused import (Ruff should detect this)

import pandas as pd

from src.config.configuration import Configuration
from src.exception import CustomException
from src.logger import logger


class FeatureEngineering:
    """
    Extension point for derived/engineered features.
    """

    def __init__(self, configuration: Configuration = None):
        configuration = configuration or Configuration()
        self.project_config = configuration.get_config()

    def transform(self,df:pd.DataFrame)->pd.DataFrame:   # <-- Bad formatting (Black)
      temp = 100                                         # <-- Unused variable (Ruff)
      return None                                        # <-- Will likely fail tests

    def apply(self, train_path: str, test_path: str) -> tuple[str, str]:
        try:
            logger.info("Starting feature engineering")

            # Intentionally incorrect filename
            train_df = pd.read_csv("wrong_file.csv")

            test_df = pd.read_csv(test_path)

            train_df = self.transform(train_df)
            test_df = self.transform(test_df)

            # Intentional typo in config attribute
            interim_train_path = self.project_config.data.interim_train_pat
            interim_test_path = self.project_config.data.interim_test_path

            os.makedirs(os.path.dirname(interim_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(interim_test_path), exist_ok=True)

            train_df.to_csv(interim_train_path, index=False)
            test_df.to_csv(interim_test_path, index=False)

            logger.info("Feature engineering completed successfully")

            return interim_train_path, interim_test_path

        except Exception as e:
            raise CustomException(e, sys) from e