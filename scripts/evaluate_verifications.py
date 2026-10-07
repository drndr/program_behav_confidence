"""Join a self-verification run to the correctness of the answers it judged; no model calls.

    python scripts/evaluate_verifications.py data/verifications/RUN.jsonl

The judged prediction run is found through the verification prompt files the run used,
matched by hash under data/prompts/verification/, and correctness comes from that run's
evaluation in data/evaluations/. Each row records the verdict, the stated confidence
(0-100) and, when the run stored token log-probabilities, the probability given to
"correct". A judgment counts only with both a valid verdict and a valid confidence.
Results go to data/evaluations/RUN_verdicts.jsonl, with a summary.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import sha256_file
from runs import load_run
from scoring import SCORING_MODES, parse_confidence, parse_verdict, verdict_probability


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def judged_run(manifest, data_dir):
    """Return the judged prediction run and the attempt judged per (dataset, id)."""
    runs, attempts = set(), {}
    for name, digest in manifest["configuration"]["prompt_files"].items():
        matches = [p for p in (data_dir / "prompts/verification").glob(f"*/{name}") if sha256_file(p) == digest]
        if len(matches) != 1:
            raise ValueError(f"{name}: expected one verification prompt file with this run's hash, found {len(matches)}")
        record = json.loads(matches[0].with_name(matches[0].stem + ".manifest.json").read_text(encoding="utf-8"))
        runs.add(record["prediction_run"]["path"])
        for line in matches[0].read_text(encoding="utf-8").splitlines():
            prompt = json.loads(line)
            attempts[(prompt["dataset"], prompt["id"])] = prompt["prediction_attempt"]
    if len(runs) != 1:
        raise ValueError(f"Verification prompts come from {len(runs)} prediction runs, expected one")
    return data_dir.parent / runs.pop(), attempts


def judge(row):
    """Status, verdict and confidence of one verification response."""
    if row["status"] != "completed":
        return "request_" + row["status"], None, None
    text = row.get("text") or ""
    try:
        return "judged", parse_verdict(text), parse_confidence(text)
    except ValueError:
        return "invalid_judgment", None, None


def evaluate(path, data_dir=ROOT / "data", out_dir=None):
    path, data_dir = Path(path), Path(data_dir)
    manifest, latest = load_run(path)
    prediction_run, attempts = judged_run(manifest, data_dir)
    scores_path = data_dir / "evaluations" / f"{prediction_run.stem}_scores.jsonl"
    truth = {}
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        truth.setdefault((row["dataset"], row["id"], row["repetition"]), {"attempt": row["attempt"]})[row["mode"]] = row["correct"]
    results = []
    for (dataset, identifier, repetition), row in latest.items():
        answer = truth[(dataset, identifier, repetition)]
        if answer["attempt"] != attempts[(dataset, identifier)]:
            raise ValueError(f"{dataset} {identifier}: the evaluation scored attempt {answer['attempt']} but attempt "
                             f"{attempts[(dataset, identifier)]} was judged; re-evaluate the prediction run")
        status, verdict, confidence = judge(row)
        usage = row.get("usage") or {}
        results.append({"dataset": dataset, "id": identifier, "repetition": repetition, "attempt": row.get("attempt", 1),
                        "status": status, "verdict": verdict, "confidence": confidence,
                        "p_correct": verdict_probability(row.get("text") or "", row.get("logprobs")) if status == "judged" else None,
                        **{f"correct_{mode}": answer[mode] for mode in SCORING_MODES},
                        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                        "completion_tokens": usage.get("completion_tokens")})
    summary = {}
    for dataset in sorted({r["dataset"] for r in results}):
        group = [r for r in results if r["dataset"] == dataset]
        judged = [r for r in group if r["status"] == "judged"]
        summary[dataset] = {"total": len(group), "statuses": dict(Counter(r["status"] for r in group)),
                            "with_p_correct": sum(r["p_correct"] is not None for r in judged),
                            "verdict_correct": sum(r["verdict"] == "correct" for r in judged),
                            "verdict_matches_normalized": sum((r["verdict"] == "correct") == r["correct_normalized"] for r in judged)}
    out_dir = Path(out_dir) if out_dir else data_dir / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{path.stem}_verdicts.jsonl"
    output.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    hashed = [path, scores_path, ROOT / "src/scoring.py", Path(__file__)]
    (out_dir / f"{path.stem}_verdicts.summary.json").write_text(json.dumps(
        {"verification_run": relative(path), "prediction_run": relative(prediction_run), "summary": summary,
         "sha256": {relative(p): sha256_file(p) for p in hashed}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("verification_run", type=Path, help="e.g. data/verifications/RUN.jsonl")
    try:
        evaluate(parser.parse_args().verification_run)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
