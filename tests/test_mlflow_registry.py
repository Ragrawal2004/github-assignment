import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.registry import MLflowRegistryBackend


def test_mlflow_backend_logs_model_metrics_and_registers_model(tmp_path):
    import mlflow

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    backend = MLflowRegistryBackend(
        tracking_uri=tracking_uri,
        experiment_name="test_experiment",
        registered_model_name="test_model",
    )

    model = LogisticRegression()
    model.fit(np.array([[0], [1], [0], [1]]), np.array([0, 1, 0, 1]))

    model_path = str(tmp_path / "model.pkl")
    metrics_path = str(tmp_path / "metrics.json")

    with mlflow.start_run() as run:
        backend.save_model(model, model_path)
        backend.save_json({"accuracy": 0.9, "roc_auc": 0.85}, metrics_path)
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    finished_run = client.get_run(run_id)

    assert finished_run.data.metrics["accuracy"] == 0.9
    assert finished_run.data.metrics["roc_auc"] == 0.85

    registered_versions = client.search_model_versions("name='test_model'")
    assert len(registered_versions) == 1

    # Local copies are still written regardless of backend, so
    # ModelEvaluation/PredictionPipeline work unchanged either way.
    loaded_model = backend.load_model(model_path)
    assert (
        loaded_model.predict(np.array([[1]])).tolist()
        == model.predict(np.array([[1]])).tolist()
    )

    loaded_metrics = backend.load_json(metrics_path)
    assert loaded_metrics["accuracy"] == 0.9


def test_mlflow_backend_skips_mlflow_logging_without_active_run(tmp_path):
    """save_model/save_json still write local copies even with no active run."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    backend = MLflowRegistryBackend(
        tracking_uri=tracking_uri,
        experiment_name="test_experiment",
        registered_model_name="test_model",
    )

    model = LogisticRegression()
    model.fit(np.array([[0], [1]]), np.array([0, 1]))
    model_path = str(tmp_path / "no_run_model.pkl")

    backend.save_model(model, model_path)

    loaded = backend.load_model(model_path)
    assert (
        loaded.predict(np.array([[1]])).tolist()
        == model.predict(np.array([[1]])).tolist()
    )
