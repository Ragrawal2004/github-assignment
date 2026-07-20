import pytest
from sklearn.linear_model import LogisticRegression

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.feature_engineering import FeatureEngineering
from src.models.tuner import ModelTuner


def _get_transformed_data():
    train_path, test_path = DataIngestion().initiate_data_ingestion()
    interim_train, interim_test = FeatureEngineering().apply(train_path, test_path)
    return DataTransformation().initiate_data_transformation(
        interim_train, interim_test
    )


def test_tuner_returns_fitted_model_for_known_model_name():
    X_train, X_test, y_train, y_test, _ = _get_transformed_data()

    tuner = ModelTuner()
    tuned_model = tuner.tune(
        "logistic_regression", LogisticRegression(), X_train, y_train
    )

    # A fitted model can predict without raising.
    predictions = tuned_model.predict(X_test)
    assert len(predictions) == len(y_test)


def test_tuner_falls_back_to_plain_fit_for_unknown_model_name():
    X_train, X_test, y_train, y_test, _ = _get_transformed_data()

    tuner = ModelTuner()
    model = LogisticRegression()
    fitted = tuner.tune("some_model_not_in_yaml", model, X_train, y_train)

    assert fitted is model
    predictions = fitted.predict(X_test)
    assert len(predictions) == len(y_test)


@pytest.mark.slow
def test_tune_all_returns_one_fitted_model_per_candidate():
    X_train, X_test, y_train, y_test, _ = _get_transformed_data()

    from src.models.model_factory import ModelFactory

    models = ModelFactory.get_models()
    tuner = ModelTuner()
    tuned = tuner.tune_all(models, X_train, y_train)

    assert set(tuned.keys()) == set(models.keys())
