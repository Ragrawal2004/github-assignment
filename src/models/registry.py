import os
import sys
from abc import ABC, abstractmethod
from typing import Any

from src.config.configuration import Configuration
from src.exception import CustomException
from src.logger import logger
from src.utils import load_json, load_object, save_json, save_object


class ModelRegistryBackend(ABC):
    """
    Storage contract for the model registry. `ModelRegistry` talks only
    to this interface, never to joblib/MLflow/boto3/etc. directly.
    Swapping local disk storage for MLflow Model Registry, S3, or Azure
    Blob later means writing one new class here and pointing
    `registry.backend` in config.yaml at it -- `ModelTrainer`,
    `ModelSelector`, and the pipelines never change.
    """

    @abstractmethod
    def save_model(self, model: Any, path: str) -> None: ...

    @abstractmethod
    def load_model(self, path: str) -> Any: ...

    @abstractmethod
    def save_json(self, data: dict[str, Any], path: str) -> None: ...

    @abstractmethod
    def load_json(self, path: str) -> dict[str, Any]: ...


class LocalJoblibRegistryBackend(ModelRegistryBackend):
    """Default backend: models on local disk via joblib, metadata as JSON."""

    def save_model(self, model: Any, path: str) -> None:
        save_object(file_path=path, obj=model)

    def load_model(self, path: str) -> Any:
        return load_object(path)

    def save_json(self, data: dict[str, Any], path: str) -> None:
        save_json(file_path=path, data=data)

    def load_json(self, path: str) -> dict[str, Any]:
        return load_json(path)


class MLflowRegistryBackend(ModelRegistryBackend):
    """
    Logs models/metrics/metadata to an MLflow tracking server AND the
    MLflow Model Registry, while still writing local copies via joblib
    so ModelEvaluation and PredictionPipeline keep working unchanged
    regardless of which registry backend is active.

    Requires an MLflow run to be active (see src.tracking.mlflow_run) --
    if one isn't, model/metric logging to MLflow is skipped and only
    the local copy is written, so this backend degrades gracefully
    rather than failing pipeline stages that don't open a run.
    """

    def __init__(
        self, tracking_uri: str, experiment_name: str, registered_model_name: str
    ):
        import mlflow
        import mlflow.sklearn
        import mlflow.xgboost

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.registered_model_name = registered_model_name

    def save_model(self, model: Any, path: str) -> None:
        save_object(file_path=path, obj=model)

        if self._mlflow.active_run() is not None:
            self._log_model_to_mlflow(model)

    def _log_model_to_mlflow(self, model: Any) -> None:
        # XGBoost models are logged with the xgboost flavor, not the
        # generic sklearn flavor -- MLflow's sklearn flavor serializes
        # via skops, which refuses to trust xgboost's own types by
        # default. Every other model here (LogisticRegression,
        # DecisionTree, RandomForest) is a plain sklearn estimator.
        model_module = type(model).__module__
        if model_module.startswith("xgboost"):
            self._mlflow.xgboost.log_model(
                xgb_model=model,
                name="model",
                registered_model_name=self.registered_model_name,
            )
        else:
            self._mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=self.registered_model_name,
            )

    def load_model(self, path: str) -> Any:
        return load_object(path)

    def save_json(self, data: dict[str, Any], path: str) -> None:
        save_json(file_path=path, data=data)

        if self._mlflow.active_run() is not None:
            numeric_items = {
                k: float(v) for k, v in data.items() if isinstance(v, (int, float))
            }
            if numeric_items:
                self._mlflow.log_metrics(numeric_items)
            self._mlflow.log_dict(data, os.path.basename(path))

    def load_json(self, path: str) -> dict[str, Any]:
        return load_json(path)


_BACKEND_FACTORIES: dict[str, Any] = {
    "local": lambda cfg: LocalJoblibRegistryBackend(),
    "mlflow": lambda cfg: MLflowRegistryBackend(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        registered_model_name=cfg.mlflow.registered_model_name,
    ),
}


class ModelRegistry:
    """
    Single point of responsibility for persisting and retrieving the
    trained model plus its metrics and metadata. Neither ModelTrainer
    nor the pipelines write to `artifacts/` directly -- they all go
    through this class.
    """

    def __init__(
        self,
        configuration: Configuration = None,
        backend: ModelRegistryBackend = None,
    ):
        configuration = configuration or Configuration()
        self.project_config = configuration.get_config()

        if backend is not None:
            self.backend = backend
        else:
            backend_name = self.project_config.registry.backend
            factory = _BACKEND_FACTORIES.get(backend_name)
            if factory is None:
                raise CustomException(
                    f"Unsupported registry backend '{backend_name}'. "
                    f"Supported: {list(_BACKEND_FACTORIES)}",
                    sys,
                )
            self.backend = factory(self.project_config)

    def save_model(self, model: Any) -> str:
        try:
            path = self.project_config.artifacts.model_path
            self.backend.save_model(model, path)
            logger.info(f"Model saved to registry at: {path}")
            return path
        except Exception as e:
            raise CustomException(e, sys) from e

    def load_model(self) -> Any:
        try:
            return self.backend.load_model(self.project_config.artifacts.model_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def save_metrics(self, metrics: dict[str, Any]) -> str:
        try:
            path = self.project_config.artifacts.metrics_path
            self.backend.save_json(metrics, path)
            logger.info(f"Metrics saved to registry at: {path}")
            return path
        except Exception as e:
            raise CustomException(e, sys) from e

    def save_metadata(self, metadata: dict[str, Any]) -> str:
        try:
            path = self.project_config.artifacts.metadata_path
            self.backend.save_json(metadata, path)
            logger.info(f"Metadata saved to registry at: {path}")
            return path
        except Exception as e:
            raise CustomException(e, sys) from e
