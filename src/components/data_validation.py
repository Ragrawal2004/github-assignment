import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.schemas.loan_schema import LoanSchema


class DataValidation:

    def __init__(self, train_path: str):
        self.train_path = train_path

    def validate(self):
        try:

            logger.info("Reading training dataset")

            df = pd.read_csv(self.train_path)

            logger.info("Validating dataset using Pandera")

            LoanSchema.validate(df)

            logger.info("Dataset validation successful")

            return df

        except Exception as e:
            logger.error("Dataset validation failed")
            raise CustomException(e, sys) from e
