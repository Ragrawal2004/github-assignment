import os

from src.components.data_ingestion import DataIngestion


def test_data_ingestion_creates_train_and_test_files():
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
