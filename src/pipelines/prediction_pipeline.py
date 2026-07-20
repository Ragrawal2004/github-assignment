import sys
from typing import Any

import pandas as pd

from src.config.configuration import Configuration
from src.exception import CustomException
from src.logger import logger
from src.models.registry import ModelRegistry
from src.utils import load_object


class PredictionPipeline:
    """
    Single-record (or small-batch) inference:
    load model -> load preprocessor -> transform input -> predict -> return.

    Model and preprocessor are loaded once at construction time and
    reused across calls to `predict`, so this class is meant to be
    instantiated once per process (e.g. once per API worker), not once
    per request.
    """

    def __init__(
        self,
        configuration: Configuration = None,
        registry: ModelRegistry = None,
    ):
        try:
            configuration = configuration or Configuration()
            self.project_config = configuration.get_config()
            registry = registry or ModelRegistry(configuration=configuration)

            logger.info("Loading model and preprocessor for inference")
            self.model = registry.load_model()
            self.preprocessor = load_object(
                self.project_config.artifacts.preprocessor_path
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(
        self, input_data: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame
    ) -> list[Any]:
        try:
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            else:
                df = input_data

            transformed = self.preprocessor.transform(df)
            predictions = self.model.predict(transformed)

            return predictions.tolist()
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict_proba(
        self, input_data: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame
    ) -> list[list[float]]:
        try:
            if not hasattr(self.model, "predict_proba"):
                raise CustomException(
                    f"Model {type(self.model).__name__} does not support "
                    "probability predictions",
                    sys,
                )

            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            else:
                df = input_data

            transformed = self.preprocessor.transform(df)
            return self.model.predict_proba(transformed).tolist()
        except Exception as e:
            raise CustomException(e, sys) from e
