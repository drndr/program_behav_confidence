# Datasets

Raw files are local; ignore rules exclude new raw artifacts. The restored CRUXEval CSV is still tracked by the pre-rebuild Git index; ignore rules do not untrack existing files, and this step does not change the index. RE2 remains ignored. Documentation and provenance manifests are intended for version control. The initial files were copied byte-for-byte from the pre-rebuild archive; they were not freshly downloaded or revalidated against upstream.

| Dataset | Local raw file | Current use |
|---|---|---|
| CRUXEval | `raw/cruxeval/cruxeval.csv` | Load the original local code/ID/input/output table; preserve text values |
| RE2 historical export | `raw/re2/re2_bench_lite_output_prediction_prompts.jsonl` | Inspect the previous prompt-based import; not a validated experiment set |

`provenance.json` records archive source paths, destination paths, hashes, counts, and restoration date. A fresh clone needs any untracked raw artifacts supplied separately; the archive is not distributed with Git.

## Boundaries for this first step

- Loaders read data only. They do not run programs, evaluate input expressions, score predictions, or repair reference answers.
- CRUXEval inputs and outputs remain strings, including multiple-argument inputs and empty strings.
- RE2 retains complete messages, raw target representations, upstream identifiers, and metadata. Do not infer a clean code/input boundary from its prompt without a separate reviewed adapter.
- The RE2 export's recorded upstream commit is historical provenance, not a fresh verification. Previous audit findings remain in the archive and must inform future eligibility rules.
- A successful schema check does not certify benchmark correctness or permission to redistribute raw data.

## Next dataset decisions

Confirm dataset versions and provenance; inspect representative records in the notebook; then agree on common identifiers, RE2 inclusion/repair rules, and scoring semantics. Store any processed dataset separately under `processed/` with source hashes and a named transformation version. Do not overwrite raw files.
