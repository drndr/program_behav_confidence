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
python scripts/evaluate_predictions.py data/predictions/gpt54-nano-pilot.jsonl
```

Every completed response is scored right or wrong in every mode; only requests
that failed or were truncated are unscored. The modes are nested and differ in
how much is normalized before comparing. Strict compares typed Python literals
(RE2 predictions pass through `json.dumps`/`loads` like the references, so
tuples become lists and dict keys strings); object references such as
`array([1., 2.])` must be reproduced exactly as a quoted string. Relaxed also
accepts an unquoted answer as string content, ignoring surrounding line breaks
but keeping spaces. Normalized additionally applies RE2 representation rules
keyed on the reference text. Numeric/boolean array displays compare shape and
exact values, ignoring dtype and formatting. `prettyForm` displays compare their
content only (set displays ignore element order and quotes). Literals JSON could
not store, such as complex numbers, compare by value. Every other object repr or
expression is compared as a syntax tree: positional arguments are content and
must match exactly, keywords in the reference must match, and keywords the
prediction adds are taken as spelled-out defaults (a repr omits parameters at
their defaults) — except for Python builtins and standard collections such as
`dict` or `Counter`, whose keywords are content and must match — recursively; whitespace and redundant parentheses do not
matter, algebraic equivalence is not tested, and a plain value stored as text
keeps its type distinction from strict. Nothing is executed. The evaluator
refuses a run whose prompt files or source CSVs differ from the current ones.
Results go to `data/evaluations/`, using each item's latest attempt.

When token log-probabilities are available, evaluation also reports the mean
log-probability, geometric mean probability, and token count for the prediction
content, plus coverage counts in the summary. These are token-level confidence
signals, not calibrated probabilities of correctness.
