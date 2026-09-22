"""Score a saved run in all modes; no model calls. Usage: python scripts/evaluate_predictions.py RUN.jsonl

Before scoring, the run's manifest must name prompt files whose hashes equal the
current data/prompts files, and those prompts must come from the current CSVs.
Every completed response is scored right or wrong; requests that failed or were
truncated have no answer and are counted as unscored. When the run stored token
log-probabilities, the mean over the prediction's tokens is added per item.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datasets import DATASET_SOURCES, load_examples, sha256_file
from runs import load_run
from scoring import SCORING_MODES, prediction_logprobs, score_prediction

SOURCES = {name: ROOT / "data" / path for name, path in DATASET_SOURCES.items()}


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def check_provenance(run_manifest):
    """Refuse to score a run whose prompts differ from the current prompt files and CSVs."""
    for name, digest in run_manifest["configuration"]["prompt_files"].items():
        prompt_file = ROOT / "data/prompts" / name
        if not prompt_file.exists() or sha256_file(prompt_file) != digest:
            raise ValueError(f"{name} differs from the prompts this run used; regenerate or keep the old prompts")
        manifest = json.loads(prompt_file.with_name(prompt_file.stem + ".manifest.json").read_text(encoding="utf-8"))
        dataset = manifest["dataset"]
        if sha256_file(SOURCES[dataset]) != manifest["source_sha256"]:
            raise ValueError(f"{SOURCES[dataset].name} differs from the CSV the {dataset} prompts were generated from")


def evaluate(path, out_dir):
    run_manifest, latest = load_run(path)
    check_provenance(run_manifest)
    examples = {(ds, r["id"]): r for ds, source in SOURCES.items() for r in load_examples(source, ds)}
    results, summary = [], {}
    for (ds, identifier, repetition), row in latest.items():
        example = examples[(ds, identifier)]
        completed = row["status"] == "completed"
        logprob = prediction_logprobs(row.get("text") or "", row.get("logprobs")) if completed else None
        for mode in SCORING_MODES:
            score = (score_prediction(example, row.get("text") or "", mode) if completed
                     else {"status": "request_" + row["status"], "correct": None})
            results.append({"dataset": ds, "id": identifier, "repetition": repetition,
                            "attempt": row.get("attempt", 1), "mode": mode,
                            "prediction": row.get("text"), "reference": example["output"], **score,
                            "prediction_mean_logprob": logprob["mean_logprob"] if logprob else None,
                            "prediction_geometric_mean_probability": logprob["geometric_mean_probability"] if logprob else None,
                            "prediction_logprob_tokens": logprob["token_count"] if logprob else None})
    for ds in sorted({r["dataset"] for r in results}):
        for mode in SCORING_MODES:
            group = [r for r in results if r["dataset"] == ds and r["mode"] == mode]
            summary[f"{ds}/{mode}"] = {
                "total": len(group), "correct": sum(r["correct"] is True for r in group),
                "incorrect": sum(r["correct"] is False for r in group),
                "unscored": sum(r["correct"] is None for r in group),
                "with_logprobs": sum(r["prediction_mean_logprob"] is not None for r in group),
                "statuses": dict(Counter(r["status"] for r in group))}
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{path.stem}_scores.jsonl"
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    hashed = [path, *SOURCES.values(), ROOT / "src/scoring.py", Path(__file__)]
    record = {"run": relative(path), "sha256": {relative(p): sha256_file(p) for p in hashed}, "summary": summary}
    output.with_suffix(".summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/evaluations")
    args = parser.parse_args()
    try:
        evaluate(args.predictions, args.out_dir)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
