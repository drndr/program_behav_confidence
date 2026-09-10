# Program behaviour prediction: accuracy, confidence, and program properties

Research project targeting an FSE 2027 submission. Rebuild started on 10 September 2026.

## Research question

Which program and input properties are associated with output-prediction mistakes, and does post-hoc verifier confidence reflect those difficulties?

## Current state

The previous project is preserved in the local [pre-rebuild archive](archive/2026-09-10_pre-rebuild/README.md). Git history remains intact. The archive includes uncommitted and ignored files and is intentionally excluded from Git; it will not accompany a fresh clone.

The rebuild starts with dataset loading and inspection. No old runner, evaluation policy, dependency environment, or experiment configuration is automatically adopted.

## Layout

| Directory | Purpose |
|---|---|
| `src/behavior_prediction/` | Reusable Python code, beginning with dataset loaders |
| `notebook/` | Jupyter notebooks for interactive inspection and exploration |
| `scripts/` | Thin research entry points that use the package |
| `tests/` | Unit tests for reusable code |
| `data/` | Dataset documentation, provenance, and local raw/processed datasets |

Reusable logic belongs in `src`, not duplicated in notebooks or scripts. Raw data is preserved unchanged; processed versions must have their own provenance. Model predictions will not be stored as raw datasets.

## Dataset inspection

Create a new Python environment and install this project with `python -m pip install -e ".[notebook]"`. Then run `python scripts/inspect_datasets.py` or open `notebook/01_datasets.ipynb` in Jupyter. Run unit tests with `python -m unittest discover -s tests`.

See [data/README.md](data/README.md) for the local dataset inventory and limitations. No dataset code is executed by the loaders.

## Decisions carried forward

- Focus on relationships between correctness, confidence, and program/input properties.
- Use existing benchmarks: CRUXEval and RE2, subject to explicit validity and scoring checks.
- Primary property families: control flow, operations, and input structure, with size controls and a small classical-metric comparison.
- GPT-5.4 nano results are pilot evidence only.
- Main-study model candidates: DeepSeek V4 Flash, a specified Qwen3.8 variant, Claude Sonnet or Haiku, and GPT-5.6 Terra or Luna. Exact models, endpoints, reasoning settings, and budgets remain to be selected.
- Separate answer generation from assessment of fixed answers. Provider reasoning settings are not assumed equivalent.
- Calibration generalisation remains an appendix-or-omit extension.

See [REBUILD.md](REBUILD.md) for the step-by-step sequence. The historical detailed plan is retained in the archive as a reference; it is not the frozen protocol for the rebuilt project.
