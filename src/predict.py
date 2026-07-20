"""
CLI entrypoint for single-record inference.

Usage:
    python -m src.predict '{"person_age": 25, "person_income": 55000, ...}'

This is intentionally thin -- all real logic lives in
src.pipelines.prediction_pipeline.PredictionPipeline. This file exists
because the target architecture calls for a top-level `predict.py`
convenience entrypoint (e.g. for a Dockerfile CMD or a quick manual
check), not a place for logic to live.
"""

import json
import sys

from src.exception import CustomException
from src.pipelines.prediction_pipeline import PredictionPipeline


def main() -> None:
    try:
        if len(sys.argv) != 2:
            print("Usage: python -m src.predict '{\"feature\": value, ...}'")
            sys.exit(1)

        input_data = json.loads(sys.argv[1])

        pipeline = PredictionPipeline()
        prediction = pipeline.predict(input_data)

        print(json.dumps({"prediction": prediction}))

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    main()
