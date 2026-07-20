import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.registry import ModelRegistry


def test_registry_round_trips_model_metrics_and_metadata():
    registry = ModelRegistry()

    model = LogisticRegression()
    model.fit(np.array([[0], [1], [0], [1]]), np.array([0, 1, 0, 1]))

    registry.save_model(model)
    loaded_model = registry.load_model()
    assert (
        loaded_model.predict(np.array([[1]])).tolist()
        == model.predict(np.array([[1]])).tolist()
    )

    registry.save_metrics({"accuracy": 0.93, "roc_auc": np.float64(0.88)})
    registry.save_metadata({"best_model_name": "logistic_regression"})

    saved_metrics = registry.backend.load_json(
        registry.project_config.artifacts.metrics_path
    )
    saved_metadata = registry.backend.load_json(
        registry.project_config.artifacts.metadata_path
    )

    assert saved_metrics["accuracy"] == 0.93
    assert saved_metadata["best_model_name"] == "logistic_regression"
