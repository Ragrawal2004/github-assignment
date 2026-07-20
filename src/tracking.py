import sys
from collections.abc import Iterator
from contextlib import contextmanager

from src.config.configuration import Configuration
from src.exception import CustomException


@contextmanager
def mlflow_run(configuration: Configuration) -> Iterator[None]:
    """
    Opens an MLflow run scoped to one training job, and logs the run
    params (test_size, random_state, selection_metric, search_type, cv)
    up front. When `registry.backend` isn't "mlflow" this is a no-op
    context, so `train_tune_select_and_register` doesn't need an
    if/else on the backend -- it just always wraps its work in this.
    """
    try:
        project_config = configuration.get_config()

        if project_config.registry.backend != "mlflow":
            yield
            return

        import mlflow

        mlflow.set_tracking_uri(project_config.mlflow.tracking_uri)
        mlflow.set_experiment(project_config.mlflow.experiment_name)

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "test_size": project_config.training.test_size,
                    "random_state": project_config.training.random_state,
                    "selection_metric": project_config.training.selection_metric,
                    "search_type": project_config.training.tuning.search_type,
                    "cv": project_config.training.tuning.cv,
                }
            )
            yield

    except Exception as e:
        raise CustomException(e, sys) from e
