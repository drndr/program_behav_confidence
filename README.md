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

Evaluate a saved run in strict, relaxed, and normalized modes:

```sh
python -m pip install "scikit-learn==1.7.2"
python scripts/evaluate_predictions.py data/predictions/gpt54-nano-pilot.jsonl
```

Strict compares typed Python literals; relaxed also accepts unquoted strings when
literal parsing fails. All modes require prediction tags. Unquoted strings in
relaxed/normalized modes ignore surrounding CR/LF characters, retaining spaces;
quoted strings retain all content. Exact parsed reference matches are accepted.
Normalized additionally compares RE2 numeric arrays (shape and exact values,
ignoring dtype), complex numbers, and symbolic-expression syntax, including nested
outputs. Empty-argument references for recognized scikit-learn estimators accept the same
constructor name with any arguments (class-name comparison only, not state or
parameter equivalence). Recognition uses installed `BaseEstimator` class metadata,
cached once; the library version is recorded in evaluation summaries.
Other constructors, including empty containers, do not
receive this shortcut. Unsupported boolean/string array displays remain unscored.
`prettyForm` compares display content, ignoring its wrapper and layout metadata;
set displays of names/strings also ignore element order and element quotes.
Other display text retains its whitespace. This deliberately relaxes return types.
It does not execute expressions or test algebraic equivalence. Other unknown
constructors and unsupported array displays remain unscored; object detection is
heuristic because RE2 stores these as strings. Results go to `data/evaluations/`;
summaries separate correct, incorrect, and unscored answers, using the latest attempt.
