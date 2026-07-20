## What does this PR do?

<!-- One or two sentences. If it fixes a bug, what was broken? -->

## Why

<!-- What prompted this change? Link an issue if there is one. -->

## Checklist

- [ ] `pre-commit run --all-files` passes locally (or CI's `code-quality` job is green)
- [ ] `pytest` passes locally (or CI's `tests` job is green)
- [ ] If this changes `config/config.yaml` or `config/model_params.yaml`, I ran `dvc repro` locally to confirm the pipeline still completes
- [ ] If this changes a component's public interface (constructor args, return shape), I checked every caller (components/pipelines/scripts/tests) that uses it

## What was tested manually (if anything CI can't cover)

<!-- e.g. "ran the MLflow backend end-to-end against a local server", "verified the CLI predict output looks right" -->
