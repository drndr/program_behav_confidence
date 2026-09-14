"""Read dataset artifacts without evaluating code or interpreting target values."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CruxEvalExample:
    id: str
    code: str
    input: str
    output: str


def load_cruxeval(path: str | Path) -> list[CruxEvalExample]:
    """Preserve the CSV's text exactly, including empty-string targets.

    Schema validation is not validation that the reference answer is correct.
    Inputs may represent multiple function arguments; do not coerce their text.
    """
    examples = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "code", "input", "output"}
        if reader.fieldnames is None or set(reader.fieldnames) != required or len(reader.fieldnames) != 4:
            raise ValueError("CRUXEval requires exactly id, code, input, output columns")
        for index, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CRUXEval row {index}")
            if not row["id"].strip() or not row["code"].strip():
                raise ValueError(f"Missing ID or code in row {index}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate CRUXEval ID: {row['id']}")
            seen.add(row["id"])
            examples.append(CruxEvalExample(**row))
    return examples


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


def write_re2_csv(rows, path):
    """Write already-cleaned id/code/input/output records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "code", "input", "output"])
        writer.writeheader()
        writer.writerows(rows)
