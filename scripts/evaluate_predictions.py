"""Score a saved run in all modes; no model calls. Usage: python scripts/evaluate_predictions.py RUN.jsonl"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predictions import SCORING_MODES, load_examples, score_prediction


def evaluate(path, out_dir):
    # Require the declared scoring dependency before writing any results.
    sklearn_version = importlib.metadata.version("scikit-learn")
    sources = {"cruxeval": ROOT / "data/raw/cruxeval/cruxeval.csv",
               "re2": ROOT / "data/raw/re2/re2_filtered.csv"}
    examples = {(ds, r["id"]): r for ds, source in sources.items()
                for r in load_examples(source, ds)}
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(row["dataset"], row["id"], row["repetition"])] = row
    results, summary = [], {}
    for (ds, identifier, repetition), row in latest.items():
        example = examples[(ds, identifier)]
        for mode in SCORING_MODES:
            score = (score_prediction(example, row.get("text") or "", mode)
                     if row["status"] == "completed" else
                     {"status": "request_" + row["status"], "correct": None})
            results.append({"dataset": ds, "id": identifier, "repetition": repetition,
                            "attempt": row.get("attempt", 1), "mode": mode,
                            "prediction": row.get("text"), "reference": example["output"], **score})
    for ds in sorted({r["dataset"] for r in results}):
        for mode in SCORING_MODES:
            group = [r for r in results if r["dataset"] == ds and r["mode"] == mode]
            summary[f"{ds}/{mode}"] = {
                "total": len(group), "correct": sum(r["correct"] is True for r in group),
                "incorrect": sum(r["correct"] is False for r in group),
                "unscored": sum(r["correct"] is None for r in group),
                "statuses": dict(Counter(r["status"] for r in group))}
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{path.stem}_scores.jsonl"
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
              hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [path.resolve(), *sources.values(), ROOT / "src/predictions.py", Path(__file__).resolve()]}
    output.with_suffix(".summary.json").write_text(
        json.dumps({"sha256": hashes, "scikit_learn_version": sklearn_version, "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/evaluations")
    args = parser.parse_args()
    evaluate(args.predictions, args.out_dir)
