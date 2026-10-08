"""Read dataset artifacts without evaluating code or interpreting target values."""

import csv
import hashlib
import json
from pathlib import Path
import re


DATASET_SOURCES = {"cruxeval": "raw/cruxeval/cruxeval.csv", "re2": "raw/re2/re2_filtered.csv"}


def load_examples(path: str | Path, dataset: str) -> list[dict]:
    """Read final CSVs as text, adding dataset and CRUXEval's entrypoint.

    Validate the schema and IDs without interpreting inputs or reference values.
    An empty RE2 entrypoint denotes a stdin program.
    """
    if dataset not in DATASET_SOURCES:
        raise ValueError(f"Unknown dataset: {dataset}")
    required = {"id", "code", "input", "output"}
    if dataset == "re2":
        required.add("entrypoint")
    csv.field_size_limit(10_000_000)
    examples, seen = [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != required or len(reader.fieldnames) != len(required):
            raise ValueError(f"{dataset} requires exactly {', '.join(sorted(required))} columns")
        for index, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed {dataset} row {index}")
            if not row["id"].strip() or not row["code"].strip():
                raise ValueError(f"Missing ID or code in row {index}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate {dataset} ID: {row['id']}")
            seen.add(row["id"])
            if dataset == "cruxeval":
                row["entrypoint"] = "f"
            examples.append(dict(row, dataset=dataset))
    return examples


def re2_source(item_id):
    """RE2's source_dataset label, recovered from the id (matches the RE2 export for all 446 items)."""
    if item_id.startswith("HumanEval_"):
        return "HumanEval"
    if item_id.startswith("ClassEval_"):
        return "Classeval"
    if re.match(r"(?:atcoder|codeforces)_", item_id):
        return "Avatar"
    if re.match(r"[\w.-]+__[\w.-]+-\d+@@", item_id):
        return "Swebench"
    return "Real" if "@@" in item_id else None


def load_re2_export(path: str | Path) -> list[dict]:
    """Read the local RE2 export, without repairing or normalizing it.

    This is not an upstream RE2 loader or an eligibility filter. Returned rows
    retain prompt context and target serialization; labels remain unvalidated.
    """
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid RE2 JSON on line {line_number}") from error
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
                raise ValueError(f"Missing RE2 ID on line {line_number}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate RE2 ID: {row['id']}")
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages or any(
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError(f"Invalid RE2 messages on line {line_number}")
            if "target_value" not in row or not isinstance(row.get("target_output"), str):
                raise ValueError(f"Missing RE2 target fields on line {line_number}")
            seen.add(row["id"])
            rows.append(row)
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
