# Program-Behavior Prediction Reliability

Research repository for an FSE 2027 study of confidence, calibration, selective
prediction, and calibration transport in LLM program-behavior prediction.

## Paper identity

Program-behavior benchmark accuracy does not determine whether individual
predictions can be trusted. This project separates four questions:

1. Can a post-hoc estimator distinguish correct from incorrect predictions?
2. Is its confidence calibrated?
3. Can abstention reduce operational risk?
4. Does the reliability mapping transport across evaluation environments?

The primary benchmarks are CRUXEval and RE2-Bench. The primary completed model
is GPT-5.4 Nano. One open-weight cross-family replication is planned.

## Repository layout

- `scripts/analysis/`: deterministic scoring, calibration, and transport analyses
- `scripts/inference/`: controlled OpenRouter runner and frozen configurations
- `scripts/prompt_generation/`: confidence and benchmark prompt builders
- `tests/`: focused unit tests for the submission-critical pipeline
- `research_design/`: frozen protocols, manifests, aggregate results, and the submission contract
- `literature/`: literature tables and coding notes
- `paper/`: manuscript outline and bibliography
- `_local/`: ignored raw predictions, prompts, and source PDFs

The `_local/` directory is intentionally excluded from Git. It contains raw or
licensing-sensitive working artifacts copied from the earlier research
repository. Aggregate results, hashes, protocols, and analysis code are kept in
version control.

## Setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Set `OPENROUTER_API_KEY` in the local environment before running inference. Do
not store the key in this repository.

## Scope boundary

The earlier synthetic datasets, red-black-tree experiments, R vignette probe,
parameter-scaling study, and broad benchmark-claim audit are not part of this
repository. They remain preserved in the previous repository as research
provenance.

