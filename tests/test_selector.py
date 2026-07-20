from src.config.configuration import Configuration
from src.models.selector import ModelSelector


def test_selector_picks_highest_value_for_configured_metric():
    selection_metric = Configuration().get_config().training.selection_metric

    fake_report = {
        "model_a": {"model": "A", selection_metric: 0.70},
        "model_b": {"model": "B", selection_metric: 0.91},
        "model_c": {"model": "C", selection_metric: 0.85},
    }

    selector = ModelSelector()
    best_name, best_model, best_metrics = selector.select_best(fake_report)

    assert best_name == "model_b"
    assert best_model == "B"
    assert best_metrics[selection_metric] == 0.91
