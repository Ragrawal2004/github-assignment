import pytest


@pytest.mark.slow
def test_model_trainer_returns_scored_report_for_every_candidate(train_a_model):
    report = train_a_model["training_report"]

    assert len(report) > 0
    for metrics in report.values():
        assert metrics["model"] is not None
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0
