"""Browse predictions, confidence signals and dataset properties on localhost. Usage: python explorer/serve.py [PORT]"""

import csv
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from confidence import GROUPS, stated_correctness, summarize
from datasets import DATASET_SOURCES, load_examples, re2_source
from runs import AUDIT, CONDITIONS, condition_rows, item_table, load_run

PAGE = Path(__file__).resolve().parent / "index.html"
EVALUATIONS = ROOT / "data/evaluations"
PREDICTIONS = ROOT / "data/predictions"
VERIFICATIONS = ROOT / "data/verifications"
PROPERTIES = ROOT / "data/program_properties"
PROPERTY_COUNTS = ("code_token_length", "input_token_length", "executable_lines", "helper_labels_excluded",
                   "cyclomatic_complexity", "target_cyclomatic_complexity")
PROPERTY_MEASURES = ("halstead_volume", "halstead_difficulty", "target_halstead_volume", "target_halstead_difficulty")


@lru_cache(maxsize=None)
def examples():
    return {(ds, r["id"]): r for ds, path in DATASET_SOURCES.items()
            for r in load_examples(ROOT / "data" / path, ds)}


def runs():
    return sorted(p.name.removesuffix("_scores.jsonl") for p in EVALUATIONS.glob("*_scores.jsonl"))


def views():
    """Menu entries and their runs: each study condition (its runs combined, as in the analysis),
    then evaluated runs that belong to no condition, under their own name."""
    available = set(runs())
    entries = {f"{c['model']} · {'no' if c['reasoning'] == 'none' else c['reasoning']} reasoning": c["runs"]
               for c in CONDITIONS if set(c["runs"]) <= available}
    in_conditions = {run for c in CONDITIONS for run in c["runs"]}
    return entries | {run: [run] for run in sorted(available - in_conditions)}


def reasoning_details(run):
    """Reasoning text and token count per item from the raw log, which evaluations do not keep.

    Empty when the log is absent or the run disabled reasoning; the latter skips
    reading a log that can be hundreds of megabytes.
    """
    path = PREDICTIONS / f"{run}.jsonl"
    if not path.exists():
        return {}
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if (manifest["configuration"]["parameters"].get("reasoning") or {}).get("enabled") is False:
        return {}
    details = {}
    for key, row in load_run(path)[1].items():
        try:
            reasoning = row["raw_response"]["choices"][0]["message"].get("reasoning")
        except (TypeError, KeyError, IndexError, AttributeError):
            reasoning = None
        tokens = ((row.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
        details[key] = {"reasoning": reasoning, "reasoning_tokens": tokens}
    return details


def audit():
    """The RE2 audit (runs.AUDIT) by item id: whether the item is solvable as given, its category and note.

    Read on every request, so edits to the audit show on reload.
    """
    path = ROOT / "data" / AUDIT
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["id"]: {"solvable": row["solvable"] == "True", "category": row["category"], "note": row["note"]}
                for row in csv.DictReader(stream)}


def items(run):
    """One record per item: code, input, reference, response, reasoning, answer logprob,
    reasoning tokens, correct and status per scoring mode, and the audit entry of RE2 items."""
    details = reasoning_details(run)
    audited = audit()
    grouped = {}
    with (EVALUATIONS / f"{run}_scores.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["dataset"], row["id"], row["repetition"])
            if key not in grouped:
                example = examples()[(row["dataset"], row["id"])]
                grouped[key] = {"dataset": row["dataset"], "id": row["id"], "repetition": row["repetition"],
                                "code": example["code"], "input": example["input"],
                                "reference": row["reference"], "response": row["prediction"],
                                "logprob": row["prediction_mean_logprob"],
                                "reasoning": None, "reasoning_tokens": None, **details.get(key, {}),
                                "audit": audited.get(row["id"]) if row["dataset"] == "re2" else None,
                                "correct": {}, "status": {}}
            grouped[key]["correct"][row["mode"]] = row["correct"]
            grouped[key]["status"][row["mode"]] = row["status"]
    return list(grouped.values())


def confidence_items(run):
    """Every item of a prediction run with its confidence signals (runs.item_table) and the stated
    confidence as the probability that the answer is correct, plus the code, input, reference,
    response and verifier response for the detail pane."""
    texts = {(i["dataset"], i["id"], i["repetition"]): i for i in items(run)}
    table = item_table(run, ROOT / "data")
    verification = table[0]["verification_run"] if table else None
    log = VERIFICATIONS / f"{verification}.jsonl"
    responses = load_run(log)[1] if verification and log.exists() else {}
    records = []
    for row in table:
        key = (row["dataset"], row["id"], row["repetition"])
        records.append(row | {field: texts[key][field] for field in ("code", "input", "reference", "response")}
                       | {"stated_correctness": stated_correctness(row),
                          "verifier_response": (responses.get(key) or {}).get("text")})
    return records


@lru_cache(maxsize=None)
def _summary(view_runs, stamp):
    return summarize(condition_rows(list(view_runs), ROOT / "data"),
                     groups={name: GROUPS[name] for name in ("CRUXEval", "RE2")})


def confidence_summary(view):
    """confidence.summarize for a menu entry's runs combined, cached until an evaluation file changes."""
    stamp = tuple(sorted((p.name, p.stat().st_mtime_ns) for p in EVALUATIONS.glob("*.jsonl")))
    return _summary(tuple(views()[view]), stamp)


def dataset_items(name):
    """One record per program: code, input, reference output, source and measured properties."""
    with (PROPERTIES / f"{name}_properties.csv").open(encoding="utf-8", newline="") as stream:
        properties = {row["id"]: row for row in csv.DictReader(stream)}
    executions = {}
    if (PROPERTIES / f"{name}_execution.csv").exists():
        with (PROPERTIES / f"{name}_execution.csv").open(encoding="utf-8", newline="") as stream:
            executions = {row["id"]: row for row in csv.DictReader(stream)}
    records = []
    for (dataset, item_id), example in examples().items():
        if dataset == name:
            measured = properties[item_id]
            records.append({"id": item_id, "source": re2_source(item_id) if name == "re2" else "CRUXEval",
                            "code": example["code"], "input": example["input"], "reference": example["output"],
                            **{k: int(measured[k]) if measured[k] else None for k in PROPERTY_COUNTS},
                            **{k: float(measured[k]) if measured[k] else None for k in PROPERTY_MEASURES},
                            "complexity_target": measured["complexity_target"],
                            # Trace length counts only where the run reproduced the reference output.
                            "trace_length": (int(executions[item_id]["trace_length"])
                                             if executions.get(item_id, {}).get("output_matches") == "True" else None),
                            "execution_status": executions.get(item_id, {}).get("execution_status", "not run"),
                            "line_status": measured["line_status"], "line_detail": measured["line_detail"]})
    return records


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path, entries = unquote(self.path), views()
        if path == "/":
            self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/runs":
            self.send(json.dumps(list(entries)).encode(), "application/json")
        elif path.startswith("/api/runs/") and path.removeprefix("/api/runs/") in entries:
            records = [item for run in entries[path.removeprefix("/api/runs/")] for item in items(run)]
            self.send(json.dumps(records).encode(), "application/json")
        elif path.startswith("/api/confidence/") and path.endswith("/summary") \
                and path.removeprefix("/api/confidence/").removesuffix("/summary") in entries:
            view = path.removeprefix("/api/confidence/").removesuffix("/summary")
            self.send(json.dumps(confidence_summary(view)).encode(), "application/json")
        elif path.startswith("/api/confidence/") and path.removeprefix("/api/confidence/") in entries:
            records = [item for run in entries[path.removeprefix("/api/confidence/")] for item in confidence_items(run)]
            self.send(json.dumps(records).encode(), "application/json")
        elif path == "/api/datasets":
            self.send(json.dumps(list(DATASET_SOURCES)).encode(), "application/json")
        elif path.startswith("/api/datasets/") and path.removeprefix("/api/datasets/") in DATASET_SOURCES:
            self.send(json.dumps(dataset_items(path.removeprefix("/api/datasets/"))).encode(), "application/json")
        else:
            self.send_error(404)

    def send(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Explorer on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    server.serve_forever()
