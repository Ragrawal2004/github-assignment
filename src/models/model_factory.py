from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


class ModelFactory:
    """
    Sole place in the project that knows how to instantiate a model.

    Keys are snake_case (e.g. "logistic_regression") because they are
    used as identifiers elsewhere too: as lookup keys into
    config/model_params.yaml, and as JSON/metrics keys in the model
    registry and DVCLive. XGBoost is included automatically when the
    `xgboost` package is installed and silently omitted otherwise, so
    the project runs with or without that optional dependency.
    """

    @staticmethod
    def get_models(random_state: int = 42) -> dict[str, Any]:
        models: dict[str, Any] = {
            "logistic_regression": LogisticRegression(),
            "decision_tree": DecisionTreeClassifier(random_state=random_state),
            "random_forest": RandomForestClassifier(random_state=random_state),
        }

        if _XGBOOST_AVAILABLE:
            models["xgboost"] = XGBClassifier(
                random_state=random_state, eval_metric="logloss"
            )

        return models
