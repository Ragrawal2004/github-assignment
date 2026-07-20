"""
DVC stage entrypoint: train.csv/test.csv -> interim (feature-engineered)
CSVs -> transformed arrays + fitted preprocessor.

FeatureEngineering and DataTransformation are two separate components,
but the DVC stage list requested (Phase 10) has a single
`data_transformation` stage, not a dedicated `feature_engineering`
stage -- so this script runs both in sequence.
"""

from src.components.data_transformation import DataTransformation
from src.components.feature_engineering import FeatureEngineering
from src.config.configuration import Configuration

if __name__ == "__main__":
    config = Configuration().get_config()

    interim_train_path, interim_test_path = FeatureEngineering().apply(
        config.data.train_data_path, config.data.test_data_path
    )

    DataTransformation().initiate_data_transformation(
        interim_train_path, interim_test_path
    )
    print("Data transformation completed.")
