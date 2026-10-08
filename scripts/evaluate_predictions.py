"""Score a saved run in all modes; no model calls. Usage: python scripts/evaluate_predictions.py RUN.jsonl

Every completed response is scored right or wrong; requests that failed or were
truncated have no answer and are counted as unscored. When the run stored token
log-probabilities, the mean over the prediction's tokens is added per item, as are
the generated and reasoning token counts, finish reason and backend provider.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datasets import DATASET_SOURCES, load_examples, sha256_file
from runs import load_run
from scoring import SCORING_MODES, prediction_logprobs, score_prediction

SOURCES = {name: ROOT / "data" / path for name, path in DATASET_SOURCES.items()}


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def evaluate(path, out_dir):
    _, latest = load_run(path)
    examples = {(ds, r["id"]): r for ds, source in SOURCES.items() for r in load_examples(source, ds)}
    results, summary = [], {}
    for (ds, identifier, repetition), row in latest.items():
        example = examples[(ds, identifier)]
        completed = row["status"] == "completed"
        logprob = prediction_logprobs(row.get("text") or "", row.get("logprobs")) if completed else None
        usage = row.get("usage") or {}
        request = {"finish_reason": row.get("finish_reason"), "backend_provider": row.get("backend_provider"),
                   "completion_tokens": usage.get("completion_tokens"),
                   "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        for mode in SCORING_MODES:
            score = (score_prediction(example, row.get("text") or "", mode) if completed
                     else {"status": "request_" + row["status"], "correct": None})
            results.append({"dataset": ds, "id": identifier, "repetition": repetition,
                            "attempt": row.get("attempt", 1), "mode": mode,
                            "prediction": row.get("text"), "reference": example["output"], **score,
                            "prediction_mean_logprob": logprob["mean_logprob"] if logprob else None,
                            "prediction_geometric_mean_probability": logprob["geometric_mean_probability"] if logprob else None,
                            "prediction_logprob_tokens": logprob["token_count"] if logprob else None, **request})
    for ds in sorted({r["dataset"] for r in results}):
        for mode in SCORING_MODES:
            group = [r for r in results if r["dataset"] == ds and r["mode"] == mode]
            logprobs = [r["prediction_mean_logprob"] for r in group
                        if r["prediction_mean_logprob"] is not None]
            summary[f"{ds}/{mode}"] = {
                "total": len(group), "correct": sum(r["correct"] is True for r in group),
                "incorrect": sum(r["correct"] is False for r in group),
                "unscored": sum(r["correct"] is None for r in group),
                "with_logprobs": len(logprobs),
                "mean_prediction_logprob": fmean(logprobs) if logprobs else None,
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
