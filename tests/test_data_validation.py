from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation


def test_data_validation_passes_on_ingested_data():
    train_path, _ = DataIngestion().initiate_data_ingestion()

    validator = DataValidation(train_path)
    df = validator.validate()

    assert not df.empty
    assert "loan_status" in df.columns
