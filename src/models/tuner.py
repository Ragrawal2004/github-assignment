import sys
from abc import ABC, abstractmethod
from typing import Any

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config.configuration import Configuration
from src.exception import CustomException
from src.logger import logger
from src.utils import read_yaml

# sklearn's `scoring` string doesn't always match our metric names 1:1
# (e.g. "f1_score" -> "f1"). This is the one place that mapping lives.
_SELECTION_METRIC_TO_SKLEARN_SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1_score": "f1",
    "roc_auc": "roc_auc",
}


class SearchStrategy(ABC):
    """
    One hyperparameter search algorithm. New strategies (e.g. Optuna)
    are added by writing a new subclass and registering it in
    `ModelTuner._STRATEGIES` -- `ModelTuner` itself never changes.
    """

    def __init__(self, cv: int, scoring: str, n_iter: int, random_state: int):
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state

    @abstractmethod
    def search(self, estimator: Any, param_grid: dict[str, Any], X, y) -> Any: ...


class GridSearchStrategy(SearchStrategy):
    def search(self, estimator: Any, param_grid: dict[str, Any], X, y) -> Any:
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
        )
        search.fit(X, y)
        return search.best_estimator_


class RandomizedSearchStrategy(SearchStrategy):
    def search(self, estimator: Any, param_grid: dict[str, Any], X, y) -> Any:
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            random_state=self.random_state,
        )
        search.fit(X, y)
        return search.best_estimator_


class ModelTuner:
    """
    Runs hyperparameter search for a model factory's candidates.
    Hyperparameters come from config/model_params.yaml, never hardcoded
    here; the search algorithm is selected via `training.tuning.search_type`
    in config.yaml.
    """

    _STRATEGIES: dict[str, type] = {
        "grid": GridSearchStrategy,
        "randomized": RandomizedSearchStrategy,
    }

    def __init__(
        self,
        configuration: Configuration = None,
        model_params_path: str = "config/model_params.yaml",
    ):
        configuration = configuration or Configuration()
        self.project_config = configuration.get_config()
        self.model_params: dict[str, Any] = read_yaml(model_params_path) or {}

        tuning_config = self.project_config.training.tuning
        scoring = _SELECTION_METRIC_TO_SKLEARN_SCORING.get(
            self.project_config.training.selection_metric
        )
        if scoring is None:
            raise CustomException(
                f"No sklearn scoring mapping for selection_metric "
                f"'{self.project_config.training.selection_metric}'",
                sys,
            )

        strategy_cls = self._STRATEGIES.get(tuning_config.search_type)
        if strategy_cls is None:
            raise CustomException(
                f"Unsupported search_type '{tuning_config.search_type}'. "
                f"Supported: {list(self._STRATEGIES)}",
                sys,
            )

        self.strategy: SearchStrategy = strategy_cls(
            cv=tuning_config.cv,
            scoring=scoring,
            n_iter=tuning_config.n_iter,
            random_state=self.project_config.training.random_state,
        )

    def tune(self, model_name: str, model: Any, X_train, y_train) -> Any:
        try:
            param_grid = self.model_params.get(model_name)

            if not param_grid:
                logger.info(
                    f"No hyperparameter grid for '{model_name}' in "
                    "model_params.yaml -- fitting with library defaults"
                )
                model.fit(X_train, y_train)
                return model

            logger.info(
                f"Tuning '{model_name}' with {self.strategy.__class__.__name__}"
            )
            return self.strategy.search(model, param_grid, X_train, y_train)

        except Exception as e:
            raise CustomException(e, sys) from e

    def tune_all(self, models: dict[str, Any], X_train, y_train) -> dict[str, Any]:
        try:
            return {
                name: self.tune(name, model, X_train, y_train)
                for name, model in models.items()
            }
        except Exception as e:
            raise CustomException(e, sys) from e
