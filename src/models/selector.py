import sys
from typing import Any

from src.config.configuration import Configuration
from src.exception import CustomException
from src.logger import logger


class ModelSelector:
    """
    Compares tuned/scored models and picks the best one according to
    `training.selection_metric` in config.yaml. This is the ONLY place
    in the project that compares models against each other -- neither
    ModelTrainer nor ModelTuner make that call.
    """

    def __init__(self, configuration: Configuration = None):
        configuration = configuration or Configuration()
        self.selection_metric = configuration.get_config().training.selection_metric

    def select_best(
        self, report: dict[str, dict[str, Any]]
    ) -> tuple[str, Any, dict[str, Any]]:
        try:
            if not report:
                raise ValueError("Cannot select a best model from an empty report")

            best_name = max(
                report, key=lambda name: report[name][self.selection_metric]
            )
            best_metrics = report[best_name]
            best_model = best_metrics["model"]

            logger.info(
                f"Selected '{best_name}' as best model "
                f"({self.selection_metric}={best_metrics[self.selection_metric]})"
            )

            return best_name, best_model, best_metrics

        except Exception as e:
            raise CustomException(e, sys) from e
