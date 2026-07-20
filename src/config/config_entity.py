"""
Typed configuration schema (Pydantic).

Every section of config.yaml is mirrored here as a BaseModel. This is
what buys us "configuration driven, but still type-safe": a malformed
config.yaml fails fast at startup with a clear validation error instead
of surfacing as a cryptic KeyError deep inside a pipeline stage.
"""

from typing import Literal

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    interim_train_path: str
    interim_test_path: str


class FeatureConfig(BaseModel):
    numerical_columns: list[str]
    categorical_columns: list[str]
    target_column: str


class ArtifactConfig(BaseModel):
    model_path: str
    preprocessor_path: str
    transformed_train_path: str
    transformed_test_path: str
    metrics_path: str
    metadata_path: str
    evaluation_dir: str


class TuningConfig(BaseModel):
    search_type: Literal["grid", "randomized"]
    cv: int = Field(ge=2)
    n_iter: int = Field(ge=1)


class TrainingConfig(BaseModel):
    test_size: float = Field(gt=0, le=0.3)
    random_state: int
    selection_metric: Literal[
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]
    tuning: TuningConfig


class RegistryConfig(BaseModel):
    backend: Literal["local", "mlflow"]


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    registered_model_name: str


class Config(BaseModel):
    data: DataConfig
    artifacts: ArtifactConfig
    training: TrainingConfig
    registry: RegistryConfig
    mlflow: MLflowConfig
    features: FeatureConfig
