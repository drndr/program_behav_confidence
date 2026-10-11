"""Read saved model runs, and join a run's results into one table per item."""

import csv
import json
from pathlib import Path

from datasets import re2_source

HALSTEAD_FIELDS = ("halstead_volume", "halstead_difficulty", "target_halstead_volume", "target_halstead_difficulty")

# The study: the prediction runs of each model and reasoning setting. Medium was run per
# dataset (RE2 first, CRUXEval later) with identical settings; its two runs form one condition.
CONDITIONS = [
    {"model": "GPT-6 Luna", "reasoning": "none", "runs": ["gpt6.0_luna_no_reasoning"]},
    {"model": "GPT-6 Luna", "reasoning": "low", "runs": ["gpt6.0_luna_low_reasoning"]},
    {"model": "GPT-6 Luna", "reasoning": "medium",
     "runs": ["gpt6.0_luna_medium_reasoning", "gpt6.0_luna_medium_reasoning_cruxeval"]},
]


def load_run(path):
    """Return the run manifest and its latest row per (dataset, id, repetition).

    The runner appends on retries, so a log can hold several attempts for one item.
    """
    path = Path(path)
    manifest = json.loads(path.with_name(path.stem + ".manifest.json").read_text(encoding="utf-8"))
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(row["dataset"], row["id"], row["repetition"])] = row
    return manifest, latest


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def verification_of(run, data_dir):
    """Name of the self-verification run that judged a prediction run, or None."""
    names = [p.name.removesuffix("_verdicts.summary.json")
             for p in (Path(data_dir) / "evaluations").glob("*_verdicts.summary.json")
             if Path(json.loads(p.read_text(encoding="utf-8"))["prediction_run"]).stem == run]
    if len(names) > 1:
        raise ValueError(f"{run} has {len(names)} verifications; expected at most one")
    return names[0] if names else None


def item_table(run, data_dir):
    """One row per item of a prediction run, from committed files only.

    Joins the run's evaluation (correctness per scoring mode, answer log-probability,
    token counts), its self-verification if there is one, and the program properties,
    including the trace length from running the program where it was measured.
    """
    data_dir = Path(data_dir)
    rows = {}
    for score in _read_jsonl(data_dir / "evaluations" / f"{run}_scores.jsonl"):
        key = (score["dataset"], score["id"], score["repetition"])
        row = rows.setdefault(key, {
            "dataset": score["dataset"], "id": score["id"], "repetition": score["repetition"],
            "attempt": score["attempt"], "logprob": score["prediction_mean_logprob"],
            "completion_tokens": score.get("completion_tokens"), "reasoning_tokens": score.get("reasoning_tokens"),
            "finish_reason": score.get("finish_reason"), "backend_provider": score.get("backend_provider")})
        row[f"correct_{score['mode']}"] = score["correct"]
        row[f"status_{score['mode']}"] = score["status"]
    properties, traces = {}, {}
    for dataset in sorted({dataset for dataset, _, _ in rows}):
        with (data_dir / "program_properties" / f"{dataset}_properties.csv").open(encoding="utf-8", newline="") as stream:
            for measured in csv.DictReader(stream):
                properties[(dataset, measured["id"])] = measured
        # Trace length counts only where running the program reproduced the reference output.
        execution = data_dir / "program_properties" / f"{dataset}_execution.csv"
        if execution.exists():
            with execution.open(encoding="utf-8", newline="") as stream:
                traces.update({(dataset, measured["id"]): int(measured["trace_length"])
                               for measured in csv.DictReader(stream) if measured["output_matches"] == "True"})
    verification = verification_of(run, data_dir)
    verdicts = ({(v["dataset"], v["id"], v["repetition"]): v
                 for v in _read_jsonl(data_dir / "evaluations" / f"{verification}_verdicts.jsonl")} if verification else {})
    for key, row in rows.items():
        measured = properties[key[:2]]
        verdict = verdicts.get(key, {})
        row.update({"source": re2_source(row["id"]) if row["dataset"] == "re2" else "CRUXEval",
                    "code_tokens": int(measured["code_token_length"]), "input_tokens": int(measured["input_token_length"]),
                    "executable_lines": int(measured["executable_lines"]) if measured["executable_lines"] else None,
                    "cyclomatic_complexity": int(measured["cyclomatic_complexity"]) if measured["cyclomatic_complexity"] else None,
                    "target_cyclomatic_complexity": (int(measured["target_cyclomatic_complexity"])
                                                     if measured["target_cyclomatic_complexity"] else None),
                    **{field: float(measured[field]) if measured[field] else None for field in HALSTEAD_FIELDS},
                    "information_flow": int(measured["information_flow"]) if measured["information_flow"] else None,
                    "trace_length": traces.get(key[:2]),
                    "parse_status": measured["line_status"], "verification_run": verification,
                    "verification_status": verdict.get("status"), "verdict": verdict.get("verdict"),
                    "confidence": verdict.get("confidence"), "p_correct": verdict.get("p_correct"),
                    "verifier_reasoning_tokens": verdict.get("reasoning_tokens")})
    return list(rows.values())


def condition_rows(runs, data_dir, table=item_table):
    """Items of one condition: its runs' item tables combined.

    The runs must share every setting except their prompt files (datasets), and an item may
    come from one run only.
    """
    settings = [json.loads((Path(data_dir) / "predictions" / f"{run}.manifest.json").read_text(encoding="utf-8"))
                ["configuration"] for run in runs]
    if any({k: v for k, v in s.items() if k != "prompt_files"} != {k: v for k, v in settings[0].items()
                                                                  if k != "prompt_files"} for s in settings):
        raise ValueError(f"runs {runs} differ in more than their datasets")
    rows, seen = [], set()
    for run in runs:
        for row in table(run, data_dir):
            key = (row["dataset"], row["id"], row["repetition"])
            if key in seen:
                raise ValueError(f"{key} appears in more than one run of {runs}")
            seen.add(key)
            rows.append(row)
    return rows
