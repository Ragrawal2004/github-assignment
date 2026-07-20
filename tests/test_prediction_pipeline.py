import pandas as pd
import pytest

from src.config.configuration import Configuration
from src.pipelines.prediction_pipeline import PredictionPipeline


def _sample_records(n: int = 3):
    config = Configuration().get_config()
    test_df = pd.read_csv(config.data.interim_test_path)
    target_column = config.features.target_column
    return test_df.drop(columns=[target_column]).head(n).to_dict(orient="records")


@pytest.mark.slow
def test_prediction_pipeline_predicts_single_record(train_a_model):
    record = _sample_records(1)[0]

    pipeline = PredictionPipeline()
    predictions = pipeline.predict(record)

    assert len(predictions) == 1
    assert predictions[0] in (0, 1)


@pytest.mark.slow
def test_prediction_pipeline_predicts_batch_of_records(train_a_model):
    records = _sample_records(3)

    pipeline = PredictionPipeline()
    predictions = pipeline.predict(records)

    assert len(predictions) == 3
    assert all(p in (0, 1) for p in predictions)
