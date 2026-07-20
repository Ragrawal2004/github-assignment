import sys
from typing import Any

from src.config.configuration import Configuration
from src.exception import CustomException
from src.logger import logger
from src.models.model_factory import ModelFactory
from src.models.tuner import ModelTuner
from src.utils import score_models


class ModelTrainer:
    """
    Orchestrates "get candidates -> tune -> score" for every model
    ModelFactory produces.

    What this class deliberately does NOT do:
    - instantiate models (ModelFactory's job)
    - compare models against each other (ModelSelector's job)
    - persist the winning model (ModelRegistry's job)

    That split is what phases 4/6/7 of the refactor asked for: each of
    those responsibilities lives in exactly one class.
    """

    def __init__(
        self,
        configuration: Configuration = None,
        tuner: ModelTuner | None = None,
    ):
        configuration = configuration or Configuration()
        self.project_config = configuration.get_config()
        self.tuner = tuner or ModelTuner(configuration=configuration)

    def initiate_model_trainer(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        live: Any = None,
    ) -> dict[str, dict[str, Any]]:
        try:
            logger.info("Starting model training")

            models = ModelFactory.get_models(
                random_state=self.project_config.training.random_state
            )

            tuned_models = self.tuner.tune_all(models, X_train, y_train)

            report = score_models(X_test=X_test, y_test=y_test, models=tuned_models)

            if live is not None:
                self._log_to_dvclive(live, report)

            return report

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def _log_to_dvclive(live: Any, report: dict[str, dict[str, Any]]) -> None:
        """Log every candidate's metrics to DVCLive, namespaced by model name."""
        for model_name, metrics in report.items():
            for metric_name in (
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ):
                value = metrics.get(metric_name)
                if value is not None:
                    live.log_metric(f"{model_name}/{metric_name}", value)
