import os
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: this runs in training/CI environments, not a GUI
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.configuration import Configuration
from src.exception import CustomException
from src.logger import logger
from src.models.registry import ModelRegistry
from src.utils import load_object, save_json


class ModelEvaluation:
    """
    Final-mile quality gate: loads the model and preprocessor that
    ModelRegistry persisted, scores them on held-out test data, and
    writes a full evaluation report to `artifacts/evaluation/` so the
    numbers survive after the run (and can be diffed across DVC runs).
    """

    def __init__(
        self,
        configuration: Configuration = None,
        registry: ModelRegistry | None = None,
    ):
        configuration = configuration or Configuration()
        self.project_config = configuration.get_config()
        self.registry = registry or ModelRegistry(configuration=configuration)

    def evaluate(self, test_path: str | None = None) -> dict[str, Any]:
        try:
            test_path = test_path or self.project_config.data.interim_test_path
            logger.info(f"Starting model evaluation using: {test_path}")

            test_df = pd.read_csv(test_path)
            target_column = self.project_config.features.target_column

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            preprocessor = load_object(self.project_config.artifacts.preprocessor_path)
            model = self.registry.load_model()

            X_test_transformed = preprocessor.transform(X_test)
            y_pred = model.predict(X_test_transformed)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_transformed)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            else:
                y_prob = None
                roc_auc = None

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc,
            }

            eval_dir = self.project_config.artifacts.evaluation_dir
            os.makedirs(eval_dir, exist_ok=True)

            save_json(os.path.join(eval_dir, "evaluation.json"), metrics)
            self._save_classification_report(eval_dir, y_test, y_pred)
            self._save_confusion_matrix(eval_dir, y_test, y_pred)
            if y_prob is not None:
                self._save_roc_curve(eval_dir, y_test, y_prob)

            logger.info(f"Model evaluation completed. Metrics: {metrics}")
            return metrics

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def _save_classification_report(eval_dir: str, y_test, y_pred) -> None:
        report_text = classification_report(y_test, y_pred, zero_division=0)
        with open(os.path.join(eval_dir, "classification_report.txt"), "w") as f:
            f.write(report_text)

    @staticmethod
    def _save_confusion_matrix(eval_dir: str, y_test, y_pred) -> None:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap="Blues")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(os.path.join(eval_dir, "confusion_matrix.png"))
        plt.close(fig)

    @staticmethod
    def _save_roc_curve(eval_dir: str, y_test, y_prob) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
        ax.set_title("ROC Curve")
        fig.tight_layout()
        fig.savefig(os.path.join(eval_dir, "roc_curve.png"))
        plt.close(fig)
