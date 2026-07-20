"""DVC stage entrypoint: validate train.csv against the Pandera schema."""

from src.components.data_validation import DataValidation
from src.config.configuration import Configuration

if __name__ == "__main__":
    train_path = Configuration().get_config().data.train_data_path
    DataValidation(train_path).validate()
    print(f"Validation passed for: {train_path}")
