import sys
from typing import Any

from dvclive import Live

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer
from src.config.configuration import Configuration
from src.exception import CustomException
from src.logger import logger
from src.models.registry import ModelRegistry
from src.models.selector import ModelSelector
from src.tracking import mlflow_run
from src.utils import load_object


def train_tune_select_and_register(
    configuration: Configuration,
    X_train,
    y_train,
    X_test,
    y_test,
    live: Live,
) -> dict[str, Any]:
    """
    Shared core of the "model_training" stage: tune every candidate,
    score them, select the best, persist it to the registry, and log
    metrics/plots to the given DVCLive run.

    Both `TrainingPipeline.run_pipeline` (full in-process run) and
    `scripts/run_model_training.py` (the standalone DVC stage) call
    this so the two never drift out of sync with each other.
    """
    project_config = configuration.get_config()

    with mlflow_run(configuration):
        report = ModelTrainer(configuration=configuration).initiate_model_trainer(
            X_train, y_train, X_test, y_test, live=live
        )

        best_name, best_model, best_metrics = ModelSelector(
            configuration=configuration
        ).select_best(report)

        registry = ModelRegistry(configuration=configuration)
        registry.save_model(best_model)
        registry.save_metrics({k: v for k, v in best_metrics.items() if k != "model"})
        registry.save_metadata(
            {
                "best_model_name": best_name,
                "selection_metric": project_config.training.selection_metric,
                "candidates_evaluated": list(report.keys()),
            }
        )

    selection_metric = project_config.training.selection_metric
    live.log_metric(
        f"selected_model/{selection_metric}", best_metrics[selection_metric]
    )
    live.log_sklearn_plot("confusion_matrix", y_test, best_model.predict(X_test))
    _log_feature_importance(
        live, best_model, project_config.artifacts.preprocessor_path
    )

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "training_report": report,
        "registry": registry,
    }


def _log_feature_importance(live: Live, model: Any, preprocessor_path: str) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    preprocessor = load_object(preprocessor_path)
    feature_names = preprocessor.get_feature_names_out()

    for name, importance in zip(feature_names, model.feature_importances_, strict=True):
        live.log_metric(f"feature_importance/{name}", float(importance))


class TrainingPipeline:
    """
    Orchestrates the full training workflow as a single call:

        DataIngestion -> DataValidation -> FeatureEngineering ->
        DataTransformation -> ModelTrainer -> ModelSelector ->
        ModelRegistry -> ModelEvaluation

    Every stage is still a standalone, independently testable component
    -- this class only sequences them and passes outputs of one stage
    as inputs to the next. It also owns the single DVCLive run for the
    whole training job, since DVCLive tracks one run's metrics, not
    one component's.
    """

    def __init__(self, configuration: Configuration = None):
        self.configuration = configuration or Configuration()
        self.project_config = self.configuration.get_config()

    def run_pipeline(self) -> dict[str, Any]:
        try:
            with Live(dir="dvclive") as live:
                logger.info("=== Training Pipeline: Data Ingestion ===")
                train_path, test_path = DataIngestion(
                    configuration=self.configuration
                ).initiate_data_ingestion()

                logger.info("=== Training Pipeline: Data Validation ===")
                DataValidation(train_path).validate()

                logger.info("=== Training Pipeline: Feature Engineering ===")
                interim_train_path, interim_test_path = FeatureEngineering(
                    configuration=self.configuration
                ).apply(train_path, test_path)

                logger.info("=== Training Pipeline: Data Transformation ===")
                X_train, X_test, y_train, y_test, preprocessor_path = (
                    DataTransformation(
                        configuration=self.configuration
                    ).initiate_data_transformation(
                        interim_train_path, interim_test_path
                    )
                )

                logger.info(
                    "=== Training Pipeline: Model Training / Selection / Registry ==="
                )
                result = train_tune_select_and_register(
                    self.configuration, X_train, y_train, X_test, y_test, live
                )

                logger.info("=== Training Pipeline: Model Evaluation ===")
                evaluation_metrics = ModelEvaluation(
                    configuration=self.configuration, registry=result["registry"]
                ).evaluate(test_path=interim_test_path)

                logger.info("Training pipeline completed successfully")

                return {
                    "best_model_name": result["best_model_name"],
                    "training_report": result["training_report"],
                    "evaluation_metrics": evaluation_metrics,
                }

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    result = TrainingPipeline().run_pipeline()
    print(f"Best model: {result['best_model_name']}")
    print(f"Evaluation metrics: {result['evaluation_metrics']}")
