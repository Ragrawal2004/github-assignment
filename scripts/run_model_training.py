"""
DVC stage entrypoint: transformed arrays -> tuned/scored candidates ->
best model selected and persisted to the registry.

Reads the transformed_train_path / transformed_test_path artifacts
that the data_transformation stage wrote, rather than recomputing
transformation itself -- keeping this a properly cacheable, independent
DVC stage. The actual tune/select/register logic lives in
`train_tune_select_and_register` (src/pipelines/train_pipeline.py) so
this script and TrainingPipeline never drift apart.
"""

from dvclive import Live

from src.config.configuration import Configuration
from src.pipelines.train_pipeline import train_tune_select_and_register
from src.utils import load_object

if __name__ == "__main__":
    configuration = Configuration()
    config = configuration.get_config()

    X_train, y_train = load_object(config.artifacts.transformed_train_path)
    X_test, y_test = load_object(config.artifacts.transformed_test_path)

    with Live(dir="dvclive") as live:
        result = train_tune_select_and_register(
            configuration, X_train, y_train, X_test, y_test, live
        )

    print(f"Best model: {result['best_model_name']}")
