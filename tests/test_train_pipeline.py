import os

import pytest

from src.config.configuration import Configuration


@pytest.mark.slow
def test_training_pipeline_runs_end_to_end_and_produces_all_artifacts(train_a_model):
    result = train_a_model

    assert "best_model_name" in result
    assert "evaluation_metrics" in result
    assert 0.0 <= result["evaluation_metrics"]["accuracy"] <= 1.0

    config = Configuration().get_config()
    assert os.path.exists(config.artifacts.model_path)
    assert os.path.exists(config.artifacts.preprocessor_path)
    assert os.path.exists(config.artifacts.metrics_path)
    assert os.path.exists(config.artifacts.metadata_path)

    eval_dir = config.artifacts.evaluation_dir
    assert os.path.exists(os.path.join(eval_dir, "evaluation.json"))
    assert os.path.exists(os.path.join(eval_dir, "classification_report.txt"))
    assert os.path.exists(os.path.join(eval_dir, "confusion_matrix.png"))
    assert os.path.exists(os.path.join(eval_dir, "roc_curve.png"))
