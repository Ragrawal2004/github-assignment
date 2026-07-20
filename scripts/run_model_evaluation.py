"""DVC stage entrypoint: evaluate the registered model on interim test data."""

from src.components.model_evaluation import ModelEvaluation

if __name__ == "__main__":
    metrics = ModelEvaluation().evaluate()
    print(f"Evaluation metrics: {metrics}")
