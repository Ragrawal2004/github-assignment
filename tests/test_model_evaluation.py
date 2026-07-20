import os

import pytest

from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import Configuration


@pytest.mark.slow
def test_model_evaluation_writes_report_files(train_a_model):
    metrics = ModelEvaluation().evaluate()

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0

    eval_dir = Configuration().get_config().artifacts.evaluation_dir
    assert os.path.exists(os.path.join(eval_dir, "evaluation.json"))
    assert os.path.exists(os.path.join(eval_dir, "classification_report.txt"))
    assert os.path.exists(os.path.join(eval_dir, "confusion_matrix.png"))
