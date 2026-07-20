import os

from src.components.data_ingestion import DataIngestion
from src.components.feature_engineering import FeatureEngineering
from src.config.configuration import Configuration


def test_feature_engineering_writes_interim_files_unchanged_shape():
    train_path, test_path = DataIngestion().initiate_data_ingestion()

    interim_train, interim_test = FeatureEngineering().apply(train_path, test_path)

    config = Configuration().get_config()
    assert interim_train == config.data.interim_train_path
    assert interim_test == config.data.interim_test_path
    assert os.path.exists(interim_train)
    assert os.path.exists(interim_test)

    import pandas as pd

    original = pd.read_csv(train_path)
    interim = pd.read_csv(interim_train)
    assert list(original.columns) == list(interim.columns)
    assert original.shape == interim.shape
