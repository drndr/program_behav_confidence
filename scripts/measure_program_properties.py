"""Measure finalized datasets: python scripts/measure_program_properties.py [--encoding o200k_base]."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tiktoken
from datasets import DATASET_SOURCES, load_examples, sha256_file
from program_properties import measure_properties

SOURCES = DATASET_SOURCES
METRICS = ("code_token_length", "input_token_length", "executable_lines",
           "cyclomatic_complexity", "target_cyclomatic_complexity", "halstead_volume", "halstead_difficulty",
           "target_halstead_volume", "target_halstead_difficulty", "information_flow")


def run(data_dir, out_dir, encoding_name="o200k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "encoding": encoding_name, "tiktoken_version": importlib.metadata.version("tiktoken"),
                "python_version": platform.python_version(), "sources": {}, "summary": {},
                "implementation_sha256": {p: sha256_file(ROOT / p) for p in
                                          ("src/program_properties.py", "scripts/measure_program_properties.py")},
                "definitions": {
                    "tokens": "Exact code/input CSV fields using encode_ordinary; full supplied code including helpers and comments, no prompt or reference. Fixed reference encoding, not provider-specific token usage.",
                    "executable_lines": "Physical code-token lines including headers/continuations; exclude blank/comment lines and standalone literal strings (docstrings). Exclude RE2 helper labels: dotted names (also dot-prefixed or with a hyphenated package) before definitions, and frame names such as pkg.mod.<genexpr>. No execution.",
                    "cyclomatic_complexity": "1 + decision points of the whole supplied code (helper labels excluded): if/elif, conditional expressions, for/while loops, comprehension for-clauses and conditions, each and/or, except clauses, match cases. No execution.",
                    "target_cyclomatic_complexity": "The same for the target alone: the entry-point function (complexity_target names it; when several functions share the name, the one defined in a class), or the whole script when there is no entry point.",
                    "halstead": "From Python tokens of the whole code (helper labels excluded): operators are operator/punctuation tokens and keywords; operands are names, numbers, strings and f-string text (a docstring is one operand); layout and comments are skipped. Volume = (operators + operands) * log2(distinct operators + distinct operands); difficulty = distinct operators / 2 * operands / distinct operands. target_halstead_* use the tokens on the target's lines.",
                    "information_flow": "Henry-Kafura information flow of the target without the length term (as IF4): (fan-in * fan-out)^2. Fan-in: parameters other than self, names read from outside the target, instance attributes read. Fan-out: distinct functions called (methods by name), instance attributes written. Calls into the target are not in the supplied code, so fan-in counts data only.",
                    "missing": "Blank executable_lines means tokenization failed, not zero. lexical_only means tokenization succeeded but AST parsing failed; source-size count only, and no complexity or Halstead measures. Blank target measures with a parsed program mean the target could not be told apart."}}
    for name, relative in SOURCES.items():
        source = data_dir / relative
        examples = load_examples(source, name)
        rows = [{"dataset": name, "id": r["id"], "tokenizer": encoding_name,
                 **measure_properties(r, encoding)} for r in examples]
        output = out_dir / f"{name}_properties.csv"
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"count": len(rows), "line_status": dict(Counter(r["line_status"] for r in rows))}
        for metric in METRICS:
            values = [r[metric] for r in rows if r[metric] is not None]
            summary[metric] = {"available": len(values), "min": min(values) if values else None,
                               "median": round(statistics.median(values), 2) if values else None,
                               "max": max(values) if values else None}
        path = source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else source.as_posix()
        manifest["sources"][name] = {"path": path, "sha256": sha256_file(source),
                                     "output_sha256": sha256_file(output)}
        manifest["summary"][name] = summary
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", default="o200k_base")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/program_properties")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, args.encoding)
