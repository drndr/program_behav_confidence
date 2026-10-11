"""Measure trace length by running each program once: python scripts/measure_execution.py [--datasets ...].

This step executes dataset code. Each program runs in its own Python process (src/execution.py)
with a timeout, a fixed hash seed, a minimal environment and a scratch working directory; the
final measurement belongs in the execution container (no network, no secrets). Trace length is
the number of line events in the program's own code. The program's actual output is compared
with the reference output: a trace counts only where the run reproduces the reference.

CRUXEval calls the entry point on its input. Of RE2 only the exercises run without their
repositories: HumanEval functions get the recorded arguments, ClassEval methods run on an
instance rebuilt from the recorded state (without calling __init__), and Avatar scripts read
the recorded standard input. RE2 records some objects as strings (a complex number as
"(-1-2j)"), so a second attempt decodes such strings; the first attempt that reproduces the
reference output is kept. RE2 library code is not executed.

Writes data/program_properties/<dataset>_execution.csv and execution_manifest.json.
"""

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import DATASET_SOURCES, load_examples, re2_source, sha256_file

CHILD = ROOT / "src/execution.py"
HASH_SEED = "0"
FIELDS = ["dataset", "id", "trace_length", "execution_status", "output_matches", "decoded_inputs", "execution_detail"]


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def decoded(value):
    """RE2 input values with strings that are Python literals of another type (e.g. "(-1-2j)") decoded."""
    if isinstance(value, dict):
        return {key: decoded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decoded(item) for item in value]
    if isinstance(value, str):
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return value
        return value if isinstance(literal, str) else literal
    return value


def harness(example):
    """Candidate payloads for an example, in order: (payload, decoded inputs). Empty without a harness."""
    code = example["code"]
    if example["dataset"] == "cruxeval":
        return [({"code": code, "call": f"{example['entrypoint']}({example['input']})"}, False)]
    source, recorded = re2_source(example["id"]), json.loads(example["input"])
    if source == "Avatar":
        stdin = recorded["input"] if recorded["input"].endswith("\n") else recorded["input"] + "\n"
        return [({"code": code, "stdin": stdin}, False)]
    if source not in ("HumanEval", "Classeval"):
        return []
    candidates = []
    for decode in (False, True):
        state, args, kwargs = (decoded(recorded[key]) if decode else recorded[key] for key in ("self", "args", "kwargs"))
        arguments = f"**{args!r}, **{kwargs!r}"
        if source == "HumanEval":
            payload = {"code": code, "call": f"{example['entrypoint']}({arguments})"}
        else:
            cls = example["id"].split("@", 1)[1].rsplit(".", 1)[0]
            payload = {"code": code, "setup": f"_self = {cls}.__new__({cls})\n_self.__dict__.update({state!r})\n",
                       "call": f"_self.{example['entrypoint']}({arguments})"}
        if all(payload != existing for existing, _ in candidates):
            candidates.append((payload, decode))
    return candidates


def same_output(result, reference):
    """Compare as Python values where both parse, else as text."""
    try:
        return ast.literal_eval(result) == ast.literal_eval(reference)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return result == reference


def output_matches(example, outcome):
    """Whether a run reproduced the reference: CRUXEval as Python values, RE2 as recorded JSON."""
    if example["dataset"] == "cruxeval":
        return same_output(outcome["result"], example["output"])
    reference = json.loads(example["output"])
    if re2_source(example["id"]) == "Avatar":
        printed = outcome["printed"].strip()
        return printed == reference.strip() if isinstance(reference, str) else same_output(printed, repr(reference))
    return outcome["result_json"] is not None and json.loads(outcome["result_json"]) == reference


def run_program(payload, timeout):
    """Run one payload in a fresh process: status, trace length, return value repr and detail."""
    environment = {"PYTHONHASHSEED": HASH_SEED, "PYTHONIOENCODING": "utf-8"}
    if "SYSTEMROOT" in os.environ:  # Windows needs it to start Python
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    with tempfile.TemporaryDirectory() as scratch:
        try:
            done = subprocess.run([sys.executable, "-s", "-P", "-B", str(CHILD)], input=json.dumps(payload),
                                  capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                                  cwd=scratch, env=environment)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "trace_length": None, "detail": f"over {timeout} s"}
    if done.returncode != 0:
        last = (done.stderr.strip().splitlines() or ["no error output"])[-1]
        return {"status": "error", "trace_length": None, "detail": last[:300]}
    return {"status": "ok", **json.loads(done.stdout.strip().splitlines()[-1]), "detail": ""}


def measure(example, timeout):
    """Run the candidates until one reproduces the reference; report it, or else the first."""
    row = {"dataset": example["dataset"], "id": example["id"], "trace_length": None,
           "execution_status": "no_harness", "output_matches": None, "decoded_inputs": None, "execution_detail": ""}
    first = None
    for payload, decode in harness(example):
        outcome = run_program(payload, timeout)
        matches = output_matches(example, outcome) if outcome["status"] == "ok" else None
        current = {**row, "trace_length": outcome["trace_length"], "execution_status": outcome["status"],
                   "output_matches": matches, "decoded_inputs": decode, "execution_detail": outcome["detail"]}
        if matches:
            return current
        first = first or current
    return first or row


def run(data_dir, out_dir, datasets, timeout=10, workers=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(),
                "platform": platform.platform(), "container_image": os.environ.get("EXECUTION_IMAGE"),
                "hash_seed": HASH_SEED, "timeout_seconds": timeout, "sources": {}, "summary": {},
                "implementation_sha256": {relative(p): sha256_file(p) for p in (CHILD, Path(__file__))},
                "definitions": {
                    "trace_length": "Line events in the program's own code while running it once on its input: each execution of a line counts, so a loop body counts once per iteration. Functions and methods: only the call is traced, not the definitions or the setup; Avatar scripts: the whole script. Use it only where output_matches is true.",
                    "output_matches": "Whether the run reproduced the reference output: CRUXEval return values as Python values, RE2 return values as recorded JSON (other objects by repr), Avatar printed output as text or number.",
                    "decoded_inputs": "RE2 only: whether the reproducing run decoded input strings that are Python literals of another type (RE2 records e.g. complex numbers as strings).",
                    "execution_status": "ok, timeout, error (the program raised; detail has the last error line) or no_harness (not executed: RE2 library code)."}}
    for name in datasets:
        source = data_dir / DATASET_SOURCES[name]
        examples = load_examples(source, name)
        with ThreadPoolExecutor(workers) as pool:
            rows = list(pool.map(lambda example: measure(example, timeout), examples))
        output = out_dir / f"{name}_execution.csv"
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        lengths = [r["trace_length"] for r in rows if r["trace_length"] is not None]
        manifest["sources"][name] = {"path": relative(source), "sha256": sha256_file(source),
                                     "output_sha256": sha256_file(output)}
        manifest["summary"][name] = {
            "count": len(rows), "execution_status": dict(Counter(r["execution_status"] for r in rows)),
            "output_matches": dict(Counter(str(r["output_matches"]) for r in rows if r["output_matches"] is not None)),
            "trace_length": {"available": len(lengths), "min": min(lengths, default=None),
                             "median": statistics.median(lengths) if lengths else None, "max": max(lengths, default=None)}}
    (out_dir / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SOURCES), default=sorted(DATASET_SOURCES))
    parser.add_argument("--timeout", type=int, default=10, help="Seconds per program")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/program_properties")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, args.datasets, args.timeout)
