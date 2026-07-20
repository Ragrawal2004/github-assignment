from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


def test_data_transformation_produces_matching_shapes():
    train_path, test_path = DataIngestion().initiate_data_ingestion()

    transformer = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = (
        transformer.initiate_data_transformation(train_path, test_path)
    )

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert preprocessor_path
