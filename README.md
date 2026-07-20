# Loan Approval MLOps

Production-grade, config-driven MLOps pipeline for a binary loan-approval
classifier. Refactored from a working-but-tangled prototype into a
Clean Architecture layout: single-responsibility components, a model
layer with no hidden coupling, DVC-orchestrated stages, and DVCLive
experiment tracking.

## Architecture

```
src/
  components/          # one pipeline stage each, no cross-talk between them
    data_ingestion.py       raw CSV -> train/test split
    data_validation.py      Pandera schema check
    feature_engineering.py  extension point for derived features (identity today)
    data_transformation.py  encode/scale -> fitted preprocessor + arrays
    model_trainer.py        tune + score every ModelFactory candidate
    model_evaluation.py     final metrics, confusion matrix, ROC curve

  models/
    model_factory.py    the only place that instantiates a model
    tuner.py             hyperparameter search (Strategy pattern: grid/randomized today, Optuna is a new class away)
    selector.py          the only place that compares models against each other
    registry.py           the only place that persists models/metrics/metadata (local joblib today, MLflow/S3/Azure = new backend class)

  pipelines/
    train_pipeline.py       orchestrates every component above, DVCLive-tracked
    prediction_pipeline.py  single/small-batch inference
    batch_pipeline.py       whole-CSV scoring, built on prediction_pipeline

  config/                # Pydantic-validated config schema + loader
  schemas/               # Pandera data schema
  logger.py, exception.py, utils.py, predict.py

config/
  config.yaml         paths, test_size, random_state, selection_metric, tuning settings, registry backend
  model_params.yaml   hyperparameter grids, keyed by ModelFactory's model names

scripts/               thin per-stage entrypoints DVC calls (no logic lives here)
dvc.yaml               data_ingestion -> data_validation -> data_transformation -> model_training -> model_evaluation
tests/                 pytest suite, one file per component/pipeline
```

## Why it's split up this way

- **ModelTrainer never instantiates a model** — `ModelFactory` does.
- **ModelTrainer never compares models** — `ModelSelector` does, using
  `training.selection_metric` from config.yaml.
- **ModelTrainer never persists anything** — `ModelRegistry` does,
  through an abstract backend so switching from local joblib to MLflow
  Registry, S3, or Azure Blob later is a new backend class, not a
  rewrite of the trainer or pipelines.
- **Nothing is hardcoded**: feature columns, target column, test size,
  random state, selection metric, search algorithm, CV folds, and
  registry backend all come from `config.yaml`; hyperparameter grids
  come from `model_params.yaml`.

## Experiment tracking

Two tracking mechanisms are wired in, controlled by `registry.backend`
in `config.yaml`:

**DVCLive (`registry.backend: local`, the default)** — every training
run writes per-candidate metrics, the winning model's confusion matrix,
and feature importances to `dvclive/`. View with `dvc exp show` /
`dvc plots show`, or the DVC VS Code extension. No server required.

**MLflow (`registry.backend: mlflow`)** — flip that one config value
and the same training run instead:
- opens a real MLflow run (params: test_size, random_state,
  selection_metric, search_type, cv)
- logs the winning model's metrics
- logs the model itself via the correct MLflow flavor (xgboost models
  use the `xgboost` flavor, everything else uses `sklearn` — the
  sklearn flavor's skops serializer otherwise refuses to trust
  XGBoost's own types)
- registers it as a new version of `loan_approval_model` in the MLflow
  Model Registry

```bash
# config/config.yaml: set registry.backend to "mlflow", then:
python -m src.pipelines.train_pipeline

# browse it:
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Local copies of the model/metrics/metadata are still written to
`artifacts/` regardless of which backend is active, so
`ModelEvaluation` and `PredictionPipeline` work unchanged either way —
switching backends never touches component code, only
`MLflowRegistryBackend` in `src/models/registry.py`.

## CI and local dev tooling

```bash
pip install -r requirements-dev.txt   # layers black/isort/ruff/pytest-cov/pre-commit on top of requirements.txt
pre-commit install                     # runs the checks below automatically on every `git commit`
```

Every push/PR runs two independent GitHub Actions jobs
(`.github/workflows/ci.yml`):

- **code-quality** — Black, isort, Ruff, YAML config validation, and an
  import smoke test. Fast (~1-2 min), no model training.
- **tests** — the full pytest suite with coverage. Slower (~4 min) —
  it genuinely trains and hyperparameter-tunes real models rather than
  mocking them.

Run the fast subset locally with `pytest -m "not slow"` (~20s) while
iterating; run the full suite (`pytest`) before pushing. `slow` and
`integration` markers are registered in `pytest.ini` for exactly this.

`.pre-commit-config.yaml` runs the same Black/isort/Ruff checks (plus
whitespace/YAML/JSON/large-file/merge-conflict hygiene checks) locally
before you commit, so CI failures on those checks should be rare.

Tool config lives in `pyproject.toml`, not scattered across
`setup.cfg`/`.flake8`/etc. Ruff is deliberately configured to **not**
sort imports itself (`I` rule disabled) — isort already owns that job,
and running both had them repeatedly re-flip each other's formatting.

CI is scaffolded (see the `FUTURE JOBS` comment block at the bottom of
`ci.yml`) for DVC pipeline validation, an MLflow smoke test, Docker
builds, security scanning, and deployment — none implemented yet, just
documented where they plug in.

## Running it

```bash
pip install -r requirements.txt

# Full pipeline, one call, DVCLive-tracked:
python -m src.pipelines.train_pipeline

# Or stage by stage via DVC (recommended — gives you caching):
dvc repro

# Single-record inference:
python -m src.predict '{"person_age": 32, "person_income": 96865, ...}'

# Batch inference:
python -m src.pipelines.batch_pipeline input.csv predictions.csv

# Tests:
pytest tests/ -v
```

## What changed from the original codebase

The reorganization also fixed real bugs found during the refactor:

1. `model_trainer.py` referenced an undefined `best_model` variable on
   the save line — every training run would have raised `NameError`.
2. Two stale duplicate copies of `data_transformation.py` and
   `model_trainer.py` existed outside `src/components/`, containing the
   same unfixed bug and bypassing `ModelFactory` entirely.
3. `requirements.txt` had code accidentally pasted into it instead of
   package names.
4. Feature column lists were duplicated across `config.yaml`,
   `schema.yaml`, and a hardcoded list in `data_transformation.py`;
   transformation now reads columns from config once.
