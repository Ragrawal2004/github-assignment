"""
Generic, reusable helper functions used across the project.

This module intentionally contains NO business logic specific to the
loan-approval domain. Anything that touches persistence (save/load),
YAML parsing, or generic model evaluation belongs here so components
never duplicate this plumbing.
"""

import json
import os
import sys
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.exception import CustomException
from src.logger import logger


def save_object(file_path: str, obj: Any) -> None:
    """Persist any Python object to disk using joblib."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path: str) -> Any:
    """Load a joblib-serialized object from disk."""
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys) from e


class _NumpyJSONEncoder(json.JSONEncoder):
    """Allows json.dump to handle numpy scalar types transparently."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(file_path: str, data: dict[str, Any]) -> None:
    """Persist a dictionary to disk as pretty-printed JSON."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            json.dump(data, file, indent=2, cls=_NumpyJSONEncoder)
        logger.info(f"JSON saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_json(file_path: str) -> dict[str, Any]:
    """Load a dictionary from a JSON file."""
    try:
        with open(file_path) as file:
            return json.load(file)
    except Exception as e:
        raise CustomException(e, sys) from e


def read_yaml(path: str) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary."""
    try:
        with open(path) as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CustomException(e, sys) from e


def _score_single_model(model: Any, X_test, y_test) -> dict[str, Any]:
    """Compute the standard classification metric set for one fitted model."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = None

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Fit every model in `models` and score it on the held-out test set.

    Use this for models that have NOT been fit yet. If your models
    already went through ModelTuner (which fits as part of the search),
    use `score_models` instead to avoid fitting twice.
    """
    try:
        report: dict[str, dict[str, Any]] = {}

        for model_name, model in models.items():
            logger.info(f"Training model: {model_name}")
            model.fit(X_train, y_train)
            report[model_name] = _score_single_model(model, X_test, y_test)

        return report
    except Exception as e:
        raise CustomException(e, sys) from e


def score_models(
    X_test,
    y_test,
    models: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Score already-fitted models on the held-out test set, without
    re-fitting them. This is what ModelTrainer uses after ModelTuner
    has already fit each candidate during hyperparameter search.
    """
    try:
        report: dict[str, dict[str, Any]] = {}

        for model_name, model in models.items():
            logger.info(f"Scoring model: {model_name}")
            report[model_name] = _score_single_model(model, X_test, y_test)

        return report
    except Exception as e:
        raise CustomException(e, sys) from e
