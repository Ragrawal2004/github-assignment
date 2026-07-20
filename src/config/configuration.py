import sys
from pathlib import Path

import yaml

from src.config.config_entity import Config
from src.exception import CustomException
from src.logger import logger


class Configuration:

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)

    def get_config(self) -> Config:
        try:
            logger.info(f"Reading configuration file: {self.config_path}")

            with open(self.config_path) as file:
                config_dict = yaml.safe_load(file)

            config = Config(**config_dict)

            logger.info("Configuration loaded successfully")

            return config

        except Exception as e:
            logger.error("Error while loading configuration")
            raise CustomException(e, sys) from e
