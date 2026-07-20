import sys

import pandas as pd

from src.config.configuration import Configuration
from src.exception import CustomException
from src.logger import logger
from src.pipelines.prediction_pipeline import PredictionPipeline


class BatchPipeline:
    """
    Scores an entire CSV of records in one pass and writes predictions
    back out to disk. Built on top of PredictionPipeline rather than
    duplicating its model/preprocessor loading logic.
    """

    def __init__(
        self,
        configuration: Configuration = None,
        prediction_pipeline: PredictionPipeline = None,
    ):
        self.configuration = configuration or Configuration()
        self.pipeline = prediction_pipeline or PredictionPipeline(
            configuration=self.configuration
        )

    def run(self, input_csv_path: str, output_csv_path: str) -> str:
        try:
            logger.info(f"Starting batch prediction on: {input_csv_path}")

            df = pd.read_csv(input_csv_path)

            target_column = self.configuration.get_config().features.target_column
            feature_df = df.drop(columns=[target_column], errors="ignore")

            predictions = self.pipeline.predict(feature_df)

            result_df = df.copy()
            result_df["prediction"] = predictions

            result_df.to_csv(output_csv_path, index=False)
            logger.info(f"Batch predictions saved to: {output_csv_path}")

            return output_csv_path

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.pipelines.batch_pipeline <input_csv> <output_csv>")
        sys.exit(1)

    BatchPipeline().run(sys.argv[1], sys.argv[2])
