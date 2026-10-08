"""Write the paper's tables from committed results; no model calls.

    python scripts/analyze.py

For every study condition (model x reasoning setting) it joins answers, confidence signals
and program properties (runs.item_table) and writes to data/analysis/:

  accuracy.csv      accuracy per dataset and scoring mode over all items, with unscored,
                    format-failure and log-probability counts
  signals.csv       per group and signal: AUROC with a 95% bootstrap interval, accuracy among
                    the most confident 80% / 50% of answers, calibration error
  properties.csv    the same within thirds of program size, per dataset
  verification.csv  self-verdicts against the answers' correctness
  analysis.json     all of the above, with the hashes of every input

Metrics follow src/confidence.py: normalized scoring; unscored answers are left out.
"""

import csv
import json
import platform
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from confidence import GROUPS, summarize
from datasets import sha256_file
from runs import item_table

# The study: one prediction run per model and reasoning setting.
CONDITIONS = [
    {"model": "GPT-6 Luna", "reasoning": "none", "run": "gpt6.0_luna_no_reasoning"},
    {"model": "GPT-6 Luna", "reasoning": "low", "run": "gpt6.0_luna_low_reasoning"},
]
MODES = ("strict", "relaxed", "normalized")
DATASETS = ("cruxeval", "re2")
SIZE_PROPERTIES = ("code_tokens", "executable_lines")
SIGNAL_FIELDS = ["model", "reasoning", "group", "items", "unscored", "accuracy", "wrong", "signal", "n",
                 "auroc", "auroc_low", "auroc_high", "accuracy_at_80", "accuracy_at_50", "ece"]


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def accuracy_rows(condition, rows):
    result = []
    for dataset in DATASETS:
        group = [row for row in rows if row["dataset"] == dataset]
        for mode in MODES:
            correct = sum(row[f"correct_{mode}"] is True for row in group)
            result.append({**condition, "dataset": dataset, "mode": mode, "items": len(group), "correct": correct,
                           "accuracy": correct / len(group) if group else None,
                           "unscored": sum(row[f"correct_{mode}"] is None for row in group),
                           "format_failures": sum(row[f"status_{mode}"] == "invalid_response" for row in group),
                           "with_logprob": sum(row["logprob"] is not None for row in group)})
    return result


def flatten(condition, summary, **extra):
    """One CSV row per group and signal."""
    return [{**condition, **extra, "group": entry["group"], "items": entry["items"], "unscored": entry["unscored"],
             "accuracy": entry["accuracy"], "wrong": entry["wrong"], "signal": signal["signal"], "n": signal["n"],
             "auroc": signal["auroc"], "auroc_low": (signal["interval"] or (None, None))[0],
             "auroc_high": (signal["interval"] or (None, None))[1],
             "accuracy_at_80": signal["selective_accuracy"]["0.8"], "accuracy_at_50": signal["selective_accuracy"]["0.5"],
             "ece": signal["ece"]}
            for entry in summary for signal in entry["signals"]]


def thirds(values):
    """Cut points at the 1/3 and 2/3 quantiles; tied values always share a third."""
    first, second = statistics.quantiles(values, n=3, method="inclusive")
    return [(min(values), first), (first, second), (second, max(values))]


def property_groups(rows):
    """Groups per dataset, size property and third: (label, dataset, property, bounds, member test)."""
    groups = []
    for dataset in DATASETS:
        for name in SIZE_PROPERTIES:
            values = [row[name] for row in rows if row["dataset"] == dataset and row[name] is not None]
            if len(set(values)) < 3:
                continue
            bounds = thirds(values)
            for index, (low, high) in enumerate(bounds):
                def member(row, dataset=dataset, name=name, index=index, bounds=bounds):
                    value = row[name]
                    return (row["dataset"] == dataset and value is not None
                            and (index == 0 or value > bounds[index - 1][1]) and (index == 2 or value <= bounds[index][1]))
                groups.append((f"{dataset} {name} third {index + 1}", dataset, name, (low, high), member))
            sizes = [sum(member(row) for row in rows) for *_, member in groups[-3:]]
            if sum(sizes) != len(values) or 0 in sizes:
                raise ValueError(f"{dataset} {name}: thirds of sizes {sizes} do not partition {len(values)} items")
    return groups


def verification_rows(condition, rows):
    result = []
    for dataset in DATASETS:
        judged = [row for row in rows if row["dataset"] == dataset and row["verdict"] is not None]
        if not judged:
            continue
        count = lambda verdict, correct: sum(row["verdict"] == verdict and row["correct_normalized"] is correct for row in judged)
        result.append({**condition, "dataset": dataset, "judged": len(judged),
                       "approved_right": count("correct", True), "approved_wrong": count("correct", False),
                       "rejected_right": count("incorrect", True), "rejected_wrong": count("incorrect", False),
                       "approval_rate": sum(row["verdict"] == "correct" for row in judged) / len(judged),
                       "accuracy": sum(row["correct_normalized"] is True for row in judged) / len(judged)})
    return result


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in rows)


def analyze(data_dir=ROOT / "data", out_dir=None, resamples=1000):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir else data_dir / "analysis"
    accuracy, signals, properties, verification, inputs = [], [], [], [], set()
    for spec in CONDITIONS:
        condition = {"model": spec["model"], "reasoning": spec["reasoning"]}
        rows = item_table(spec["run"], data_dir)
        print(f"{spec['model']}, reasoning {spec['reasoning']}: {len(rows)} items", flush=True)
        accuracy += accuracy_rows(condition, rows)
        signals += flatten(condition, summarize(rows, GROUPS, resamples))
        for label, dataset, name, (low, high), member in property_groups(rows):
            properties += flatten(condition, summarize(rows, {label: member}, resamples),
                                  dataset=dataset, property=name, low=low, high=high)
        verification += verification_rows(condition, rows)
        inputs.add(data_dir / "evaluations" / f"{spec['run']}_scores.jsonl")
        if rows and rows[0]["verification_run"]:
            inputs.add(data_dir / "evaluations" / f"{rows[0]['verification_run']}_verdicts.jsonl")
    inputs |= {data_dir / "program_properties" / f"{dataset}_properties.csv" for dataset in DATASETS}
    code = [ROOT / "src" / name for name in ("confidence.py", "metrics.py", "runs.py", "datasets.py")] + [Path(__file__)]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "accuracy.csv", accuracy)
    write_csv(out_dir / "signals.csv", signals, SIGNAL_FIELDS)
    write_csv(out_dir / "properties.csv", properties,
              SIGNAL_FIELDS[:2] + ["dataset", "property", "low", "high"] + SIGNAL_FIELDS[2:])
    if verification:
        write_csv(out_dir / "verification.csv", verification)
    record = {"conditions": CONDITIONS, "resamples": resamples, "python_version": platform.python_version(),
              "sha256": {relative(p): sha256_file(p) for p in sorted(inputs) + code},
              "accuracy": accuracy, "signals": signals, "properties": properties, "verification": verification}
    (out_dir / "analysis.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return record


def overview(record):
    """A short console view: AUROC per signal for each condition and dataset."""
    for row in record["signals"]:
        if row["group"] in ("CRUXEval", "RE2"):
            interval = f"[{row['auroc_low']:.2f}, {row['auroc_high']:.2f}]" if row["auroc_low"] is not None else ""
            print(f"  {row['model']} {row['reasoning']:4} {row['group']:8} {row['signal']:24} "
                  f"AUROC {row['auroc']:.2f} {interval:14} acc@50% {row['accuracy_at_50']:.2f}")


if __name__ == "__main__":
    overview(analyze())
