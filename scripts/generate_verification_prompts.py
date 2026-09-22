"""Write verification (confidence judge) prompts for a saved prediction run.

    python scripts/generate_verification_prompts.py data/predictions/RUN.jsonl

One prompt per completed prediction, using each item's latest attempt: the judge
sees what the predictor saw plus the candidate answer, never the reference.
Requests that failed or were truncated, and responses without a prediction tag,
have no candidate and are skipped. Output goes to data/prompts/verification/RUN/
in the runner's format, so a judge run is

    python scripts/run_predictions.py --prompt-dir data/prompts/verification/RUN --datasets re2 ...

The verification wording is not final; each manifest records wording_final.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import DATASET_SOURCES, load_examples, sha256_file
from prompt_template import verification_prompt, verification_template_sha256, verification_wording_final
from runs import load_run
from scoring import parse_prediction


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def candidates(rows):
    """Return (row, candidate text) for judgeable predictions and counts of the rest."""
    judgeable, skipped = [], Counter()
    for row in rows.values():
        if row["status"] != "completed":
            skipped["request_" + row["status"]] += 1
            continue
        try:
            judgeable.append((row, parse_prediction(row.get("text") or "")))
        except ValueError:
            skipped["no_prediction_tag"] += 1
    return judgeable, dict(skipped)


def generate(run_path, out_dir, data_dir=ROOT / "data"):
    run_path = Path(run_path)
    manifest, rows = load_run(run_path)
    if manifest["configuration"]["repetitions"] != 1:
        raise NotImplementedError("One verification prompt per candidate: multi-repetition runs are not supported yet")
    datasets = sorted({dataset for dataset, _, _ in rows})
    examples = {(name, e["id"]): e for name in datasets for e in load_examples(data_dir / DATASET_SOURCES[name], name)}
    prompts = {name: [] for name in datasets}
    judgeable, skipped = candidates(rows)
    for row, candidate in judgeable:
        example = examples[(row["dataset"], row["id"])]
        prompt = verification_prompt(example, candidate)
        prompts[row["dataset"]].append({
            "id": row["id"], "dataset": row["dataset"], "entrypoint": example["entrypoint"],
            "candidate": candidate, "prediction_run": run_path.stem, "prediction_attempt": row.get("attempt", 1),
            "messages": [{"role": "user", "content": prompt}],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for name, items in prompts.items():
        output = out_dir / f"{name}_prompts.jsonl"
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        record = {"dataset": name, "prompt_kind": "verification", "count": len(items),
                  "wording_final": verification_wording_final(),
                  "template_sha256": verification_template_sha256(),
                  "prediction_run": {"path": relative(run_path), "sha256": sha256_file(run_path),
                                     "model": manifest["configuration"]["model"],
                                     "parameters": manifest["configuration"]["parameters"]},
                  "skipped": skipped,
                  "source_path": relative(data_dir / DATASET_SOURCES[name]),
                  "source_sha256": sha256_file(data_dir / DATASET_SOURCES[name]),
                  "generated_at": datetime.now(timezone.utc).isoformat()}
        (out_dir / f"{name}_prompts.manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        manifests[name] = record
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path, help="prediction run log, e.g. data/predictions/RUN.jsonl")
    parser.add_argument("--out-dir", type=Path, help="default: data/prompts/verification/RUN")
    args = parser.parse_args()
    out_dir = args.out_dir or ROOT / "data/prompts/verification" / args.predictions.stem
    try:
        manifests = generate(args.predictions, out_dir)
    except (ValueError, FileNotFoundError, NotImplementedError) as error:
        raise SystemExit(str(error))
    for record in manifests.values():
        print(json.dumps({k: record[k] for k in ("dataset", "count", "skipped", "wording_final")}, indent=2))
    if not verification_wording_final():
        print("Verification wording is not final: prompts contain only the task, context and candidate.")


if __name__ == "__main__":
    main()
