from src.config.configuration import Configuration


def test_config_loads_successfully():
    config = Configuration().get_config()

    assert 0 < config.training.test_size <= 0.3
    assert config.data.raw_data_path.endswith(".csv")
    assert config.training.selection_metric in {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }
