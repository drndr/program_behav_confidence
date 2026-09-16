# Program Behaviour Prediction

## Repository structure

```text
src/        Reusable Python code
notebook/   Jupyter notebooks
scripts/    Research scripts
tests/      Unit tests
data/       Datasets
```

## Setup

Python 3.10+. In an activated virtual environment:

```sh
python -m pip install -e ".[notebook]"
```

## Reproduction

Generate prompts:

```sh
python scripts/generate_prompts.py
```

Preview an OpenRouter run (replace `ORGANIZATION/MODEL` with the exact model ID):

```sh
python scripts/run_predictions.py --model ORGANIZATION/MODEL --run-name pilot --limit 10 --dry-run
```

Set `OPENROUTER_API_KEY` in the root `.env` file (see `.env.example`) or in the
environment, then remove `--dry-run` to run. Environment variables take precedence.
Results are saved under `data/predictions/`. Repeat the same command to resume;
add `--retry-errors` to retry failed requests. See `--help` for repetitions,
token limits, reasoning settings, provider selection, and log-probabilities.
