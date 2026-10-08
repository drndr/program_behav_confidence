# Program Behaviour Prediction

Do code models know which programs they struggle with? Models predict program outputs
(CRUXEval, RE2-Bench-lite); we score the answers, collect confidence signals (answer
log-probability, reasoning effort, self-verification) and relate both to program properties.

## Repository structure

```text
src/        Reusable Python code
scripts/    Research scripts, one per pipeline step
tests/      Unit tests
notebook/   Jupyter notebooks
explorer/   Local results browser
data/       Datasets and results:
  raw/                  datasets (CRUXEval, filtered RE2)
  program_properties/   measured program properties
  prompts/              prediction and verification prompts
  predictions/          model runs
  verifications/        self-verification runs
  evaluations/          scores and verdicts
  analysis/             paper tables
  provenance.json, re2_*_review.json   download provenance and RE2 review decisions
```

## Setup

Python 3.12+ (program properties depend on its tokenizer). Put `OPENROUTER_API_KEY` in
`.env` (see `.env.example`) to make model calls.

```sh
python -m pip install -e ".[properties,notebook]"
```

## Pipeline

Every step writes under `data/`; the committed files already reproduce the paper's tables.

```sh
python scripts/download_datasets.py           # 1. raw datasets         -> data/raw/
python scripts/filter_re2.py                  # 2. filtered RE2 CSV     -> data/raw/re2/
python scripts/measure_program_properties.py  # 3. program properties   -> data/program_properties/
python scripts/generate_prompts.py            # 4. prediction prompts   -> data/prompts/

# 5. predictions (see --help; repeat the command to resume, --retry-errors to retry failures)
python scripts/run_predictions.py --model openai/gpt-6-luna --run-name gpt6.0_luna_no_reasoning \
    --max-tokens 16384 --backend openai --reasoning-effort none --logprobs --allow-unlisted-parameters
python scripts/evaluate_predictions.py data/predictions/RUN.jsonl           # 6. scores -> data/evaluations/

# 7. self-verification: same model and settings, judging its own answers
python scripts/generate_verification_prompts.py data/predictions/RUN.jsonl
python scripts/run_predictions.py ... --prompt-dir data/prompts/verification/RUN \
    --out-dir data/verifications --run-name RUN_self_verification
python scripts/evaluate_verifications.py data/verifications/RUN_self_verification.jsonl

python scripts/analyze.py                     # 8. paper tables         -> data/analysis/
python explorer/serve.py                      # browse everything at http://127.0.0.1:8000
```

Answers are scored in three nested modes: **strict** (typed Python literals), **relaxed**
(also unquoted strings) and **normalized** (also RE2 object representations such as arrays);
nothing is executed. See `src/scoring.py` for the rules.

Tests: `python -m unittest discover -s tests`
