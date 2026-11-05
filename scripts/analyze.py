"""Write the paper's tables from committed results; no model calls.

    python scripts/analyze.py

For every study condition (model x reasoning setting) it joins answers, confidence signals
and program properties (runs.item_table) and writes to data/analysis/:

  accuracy.csv      accuracy per group (CRUXEval, RE2, RE2 exercises, RE2 library code, and
                    RE2 and its library code without the items the audit in data/audit/ found
                    not solvable as given) and scoring mode over all items, with unscored,
                    format-failure and log-probability counts
  signals.csv       per group and signal, on every answer that has the signal: AUROC with a
                    95% bootstrap interval, the same within code-size tertiles and within source
                    (CRUXEval, RE2 exercises, RE2 repository code) and size, accuracy among the
                    most confident 80% / 50% of answers, calibration error, and the all-requests
                    view (no answer counts as wrong, no signal as least confident)
  signals_common.csv      the confidence signals and code size compared on the same answers:
                          those that have every one of them the condition provides
  properties_common.csv   the static properties compared on the same answers ("static"), and
                          with trace length on the answers that have a trace ("with trace"; RE2
                          traces exist for the exercises only)
  properties.csv    the same within thirds of each program property (size, input size,
                    cyclomatic complexity, Halstead difficulty, information flow, trace length),
                    per dataset and on RE2's solvable items, without the AUROC within size tertiles
  verification.csv  self-verdicts against the answers' correctness, per dataset and on RE2's
                    solvable items
  analysis.json     all of the above, with the hashes of every input

Metrics follow src/confidence.py: normalized scoring; unscored answers are left out except in
the all-requests view.
"""

from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
import platform
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from confidence import CONFIDENCE_SIGNALS, GROUPS, STATIC_PROPERTIES, summarize
from datasets import sha256_file
from runs import AUDIT, CONDITIONS, condition_rows

MODES = ("strict", "relaxed", "normalized")
DATASETS = ("cruxeval", "re2")
PROPERTIES = ("code_tokens", "input_tokens", "executable_lines", "cyclomatic_complexity",
              "target_cyclomatic_complexity", "halstead_difficulty", "target_halstead_difficulty",
              "information_flow", "trace_length")
SIGNAL_FIELDS = ["model", "reasoning", "group", "items", "unscored", "analysed", "accuracy", "wrong", "signal", "n",
                 "auroc", "auroc_low", "auroc_high", "within_size_tertiles_auroc", "within_size_tertiles_low",
                 "within_size_tertiles_high", "within_source_and_size_auroc", "within_source_and_size_low",
                 "within_source_and_size_high", "accuracy_at_80", "accuracy_at_50", "ece", "all_requests_n",
                 "all_requests_auroc", "all_requests_accuracy_at_80", "all_requests_accuracy_at_50"]
# Comparisons on the same answers: confidence signals against code size; properties with each other.
COMMON_SIGNALS = CONFIDENCE_SIGNALS + ("code tokens (baseline)",)
TRACED_GROUPS = {name: GROUPS[name] for name in ("CRUXEval", "RE2 exercises")}


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def accuracy_rows(condition, rows, groups=GROUPS):
    """Accuracy per group (CRUXEval, RE2 and its exercises and library code) and scoring mode, over all items."""
    result = []
    for label, member in groups.items():
        group = [row for row in rows if member(row)]
        for mode in MODES:
            correct = sum(row[f"correct_{mode}"] is True for row in group)
            result.append({**condition, "group": label, "mode": mode, "items": len(group), "correct": correct,
                           "accuracy": correct / len(group) if group else None,
                           "unscored": sum(row[f"correct_{mode}"] is None for row in group),
                           "format_failures": sum(row[f"status_{mode}"] == "invalid_response" for row in group),
                           "with_logprob": sum(row["logprob"] is not None for row in group)})
    return result


def flatten(condition, summary, **extra):
    """One CSV row per group and signal."""
    rows = []
    for entry in summary:
        for signal in entry["signals"]:
            everything = signal["all_requests"] or {"n": None, "auroc": None, "selective_accuracy": {"0.8": None, "0.5": None}}
            rows.append({**condition, **extra, "group": entry["group"], "items": entry["items"],
                         "unscored": entry["unscored"], "analysed": entry["analysed"], "accuracy": entry["accuracy"],
                         "wrong": entry["wrong"], "signal": signal["signal"], "n": signal["n"], "auroc": signal["auroc"],
                         "auroc_low": (signal["interval"] or (None, None))[0],
                         "auroc_high": (signal["interval"] or (None, None))[1],
                         "within_size_tertiles_auroc": signal["within_size_tertiles"],
                         "within_size_tertiles_low": (signal["within_size_tertiles_interval"] or (None, None))[0],
                         "within_size_tertiles_high": (signal["within_size_tertiles_interval"] or (None, None))[1],
                         "within_source_and_size_auroc": signal["within_source_and_size"],
                         "within_source_and_size_low": (signal["within_source_and_size_interval"] or (None, None))[0],
                         "within_source_and_size_high": (signal["within_source_and_size_interval"] or (None, None))[1],
                         "accuracy_at_80": signal["selective_accuracy"]["0.8"],
                         "accuracy_at_50": signal["selective_accuracy"]["0.5"], "ece": signal["ece"],
                         "all_requests_n": everything["n"], "all_requests_auroc": everything["auroc"],
                         "all_requests_accuracy_at_80": everything["selective_accuracy"]["0.8"],
                         "all_requests_accuracy_at_50": everything["selective_accuracy"]["0.5"]})
    return rows


def thirds(values):
    """Cut points at the 1/3 and 2/3 quantiles; tied values always share a third."""
    first, second = statistics.quantiles(values, n=3, method="inclusive")
    return [(min(values), first), (first, second), (second, max(values))]


def property_groups(rows):
    """Groups per dataset, and on RE2's solvable items, per property and third, with thirds cut on those items:
    (label, dataset or "re2 solvable", property, bounds, member test)."""
    scopes = [(dataset, lambda row, dataset=dataset: row["dataset"] == dataset) for dataset in DATASETS]
    scopes.append(("re2 solvable", GROUPS["RE2 solvable"]))
    groups = []
    for scope, in_scope in scopes:
        for name in PROPERTIES:
            values = [row[name] for row in rows if in_scope(row) and row[name] is not None]
            if len(set(values)) < 3:
                continue
            bounds = thirds(values)
            for index, (low, high) in enumerate(bounds):
                def member(row, in_scope=in_scope, name=name, index=index, bounds=bounds):
                    value = row[name]
                    return (in_scope(row) and value is not None
                            and (index == 0 or value > bounds[index - 1][1]) and (index == 2 or value <= bounds[index][1]))
                groups.append((f"{scope} {name} third {index + 1}", scope, name, (low, high), member))
            sizes = [sum(member(row) for row in rows) for *_, member in groups[-3:]]
            if sum(sizes) != len(values) or 0 in sizes:
                raise ValueError(f"{scope} {name}: thirds of sizes {sizes} do not partition {len(values)} items")
    return groups


def verification_rows(condition, rows):
    """Self-verdicts against correctness per dataset ("all"), and on RE2's solvable items."""
    subsets = [(dataset, "all", lambda row, dataset=dataset: row["dataset"] == dataset) for dataset in DATASETS]
    subsets.append(("re2", "solvable", GROUPS["RE2 solvable"]))
    result = []
    for dataset, subset, member in subsets:
        judged = [row for row in rows if member(row) and row["verdict"] is not None]
        if not judged:
            continue
        count = lambda verdict, correct: sum(row["verdict"] == verdict and row["correct_normalized"] is correct for row in judged)
        result.append({**condition, "dataset": dataset, "subset": subset, "judged": len(judged),
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


TABLES = ("accuracy", "signals", "signals_common", "properties_common", "properties", "verification")


def condition_tables(spec, data_dir, resamples):
    """Every table's rows for one condition, and the evaluation files they were read from."""
    condition = {"model": spec["model"], "reasoning": spec["reasoning"]}
    rows = condition_rows(spec["runs"], data_dir)
    print(f"{spec['model']}, reasoning {spec['reasoning']}: {len(rows)} items", flush=True)
    tables = {"accuracy": accuracy_rows(condition, rows),
              "signals": flatten(condition, summarize(rows, GROUPS, resamples)),
              "signals_common": flatten(condition, summarize(rows, GROUPS, resamples, signals=COMMON_SIGNALS, common=True)),
              "properties_common": (
                  flatten(condition, summarize(rows, GROUPS, resamples, signals=STATIC_PROPERTIES, common=True),
                          comparison="static")
                  + flatten(condition, summarize(rows, TRACED_GROUPS, resamples, common=True,
                                                 signals=STATIC_PROPERTIES + ("trace length (baseline)",)),
                            comparison="with trace")),
              "properties": [], "verification": verification_rows(condition, rows)}
    for label, dataset, name, (low, high), member in property_groups(rows):
        tables["properties"] += flatten(condition, summarize(rows, {label: member}, resamples, within_tertiles=False),
                                        dataset=dataset, property=name, low=low, high=high)
    inputs = {data_dir / "evaluations" / f"{run}_scores.jsonl" for run in spec["runs"]}
    inputs |= {data_dir / "evaluations" / f"{row['verification_run']}_verdicts.jsonl"
               for row in rows if row["verification_run"]}
    return tables, inputs


def analyze(data_dir=ROOT / "data", out_dir=None, resamples=1000, workers=None):
    """Conditions run in parallel processes (workers=1: one after another); every bootstrap has
    its own fixed seed, so the results do not depend on the number of workers."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir else data_dir / "analysis"
    workers = workers or min(len(CONDITIONS), os.cpu_count() or 1)
    arguments = (CONDITIONS, [data_dir] * len(CONDITIONS), [resamples] * len(CONDITIONS))
    if workers == 1:
        results = list(map(condition_tables, *arguments))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(condition_tables, *arguments))
    accuracy, signals, signals_common, properties_common, properties, verification = (
        [row for tables, _ in results for row in tables[name]] for name in TABLES)
    inputs = set().union(*(files for _, files in results))
    inputs |= {data_dir / "program_properties" / f"{dataset}_{kind}.csv" for dataset in DATASETS
               for kind in ("properties", "execution") if (data_dir / "program_properties" / f"{dataset}_{kind}.csv").exists()}
    inputs |= {data_dir / AUDIT} if (data_dir / AUDIT).exists() else set()
    code = [ROOT / "src" / name for name in ("confidence.py", "metrics.py", "runs.py", "datasets.py")] + [Path(__file__)]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "accuracy.csv", accuracy)
    write_csv(out_dir / "signals.csv", signals, SIGNAL_FIELDS)
    write_csv(out_dir / "signals_common.csv", signals_common, SIGNAL_FIELDS)
    write_csv(out_dir / "properties_common.csv", properties_common, SIGNAL_FIELDS[:2] + ["comparison"] + SIGNAL_FIELDS[2:])
    write_csv(out_dir / "properties.csv", properties,
              SIGNAL_FIELDS[:2] + ["dataset", "property", "low", "high"] + SIGNAL_FIELDS[2:])
    if verification:
        write_csv(out_dir / "verification.csv", verification)
    record = {"conditions": CONDITIONS, "resamples": resamples, "python_version": platform.python_version(),
              "sha256": {relative(p): sha256_file(p) for p in sorted(inputs) + code},
              "accuracy": accuracy, "signals": signals, "signals_common": signals_common,
              "properties_common": properties_common, "properties": properties, "verification": verification}
    (out_dir / "analysis.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return record


def overview(record):
    """A short console view: AUROC per signal for each condition and dataset."""
    for row in record["signals"]:
        if row["group"] in ("CRUXEval", "RE2"):
            interval = f"[{row['auroc_low']:.2f}, {row['auroc_high']:.2f}]" if row["auroc_low"] is not None else ""
            auroc = f"{row['auroc']:.2f}" if row["auroc"] is not None else "n/a"  # e.g. no wrong answer has the signal
            print(f"  {row['model']} {row['reasoning']:4} {row['group']:8} {row['signal']:24} "
                  f"AUROC {auroc:4} {interval:14} acc@50% {row['accuracy_at_50']:.2f}")


if __name__ == "__main__":
    overview(analyze())
