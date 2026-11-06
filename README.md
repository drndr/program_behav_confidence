# Replication Package

## Repository structure

```text
src/        Reusable Python code
scripts/    Research scripts, one per pipeline step
tests/      Unit tests
explorer/   Local results browser
server/     Notebook and instructions for the Qwen runs on a GPU server
data/       Datasets and results:
  raw/                  datasets (CRUXEval, filtered RE2)
  program_properties/   measured program properties and execution traces
  prompts/              prediction and verification prompts
  predictions/          model runs
  verifications/        self-verification runs
  evaluations/          scores and verdicts
  audit/                LLM-drafted audit of the RE2 items no condition with reasoning solved, and its review
  analysis/             analysis results behind the paper
  provenance.json, re2_*_review.json   download provenance and RE2 review decisions
```

## Setup

Python 3.12+ (program properties depend on its tokenizer). Put `OPENROUTER_API_KEY` in
`.env` (see `.env.example`) to make model calls.

```sh
python -m pip install -e ".[properties]"
```

## Pipeline

Model outputs and results are committed under `data/`, so steps 10 and 11 rerun without
model calls. The study's conditions are listed in `src/runs.py`.

1. `download_datasets.py`: download the pinned CRUXEval and RE2-Bench-lite sources.
2. `filter_re2.py`: remove the RE2 examples taken from CRUXEval and repair malformed inputs.
3. `measure_program_properties.py`: measure program size and complexity.
4. `measure_execution.py`: measure trace lengths by running the dataset programs, never model
   output; use an isolated environment.
5. `generate_prompts.py`: build the prediction prompts.
6. `run_predictions.py`: collect each condition's predictions. Qwen ran on a GPU server
   (`server/README.md`), and ten RE2 items ran later with the same settings (`*_re2_additions`).
7. `evaluate_predictions.py`: score the answers as strict, relaxed and normalized, without
   running them.
8. `generate_verification_prompts.py`, `run_predictions.py`, `evaluate_verifications.py`: have
   each model judge its own answers with the same settings, and read the verdicts.
9. `make_audit_review.py`, `audit_agreement.py`: draw the audit's review sample and compute the
   reviewers' agreement.
10. `analyze.py`: compute accuracy, signals and program properties.
11. `error_triage.py`, `paired_differences.py`: compute the fixed-budget triage and the paired
    AUROC differences.

To browse everything, run `python explorer/serve.py` and open http://127.0.0.1:8000.

Tests: `python -m unittest discover -s tests`
