import pytest

from src.pipelines.train_pipeline import TrainingPipeline


@pytest.fixture(scope="session")
def train_a_model():
    """
    Runs the full training pipeline once per test session so that
    tests needing a real, registered model (evaluation, prediction)
    don't each pay the cost of retraining -- and so they exercise the
    same artifacts a real `dvc repro` would produce.
    """
    return TrainingPipeline().run_pipeline()
