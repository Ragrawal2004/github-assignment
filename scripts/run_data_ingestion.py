"""DVC stage entrypoint: raw CSV -> train.csv / test.csv."""

from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    train_path, test_path = DataIngestion().initiate_data_ingestion()
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")
