"""Download benchmark sources and preserve their provenance."""

import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from behavior_prediction.datasets import load_cruxeval


CRUXEVAL_REPO = "cruxeval-org/cruxeval"


def cruxeval_csv(raw: bytes) -> bytes:
    """Convert upstream JSONL to CSV without interpreting Python expressions."""
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    fields = ["code", "id", "input", "output"]
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(fields):
            raise ValueError("Unexpected CRUXEval fields")
        if any(not isinstance(row[field], str) for field in fields):
            raise ValueError("CRUXEval fields must be strings")
        if not row["id"].strip() or not row["code"].strip() or row["id"] in seen:
            raise ValueError("Missing code/ID or duplicate CRUXEval ID")
        seen.add(row["id"])
    if not rows:
        raise ValueError("Empty CRUXEval download")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def download_cruxeval(data_dir: str | Path, revision: str = "main") -> dict:
    """Resolve a revision, download immutable JSONL, and save a compatible CSV."""
    data_dir = Path(data_dir)
    manifest_path = data_dir / "provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else []
    if not isinstance(manifest, list) or any(not isinstance(entry, dict) for entry in manifest):
        raise ValueError("Expected a list of provenance records")
    api_url = f"https://huggingface.co/api/datasets/{CRUXEVAL_REPO}/revision/{quote(revision, safe='')}"
    with urlopen(api_url, timeout=60) as response:
        commit = json.load(response)["sha"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Unexpected Hugging Face revision SHA")
    source_url = f"https://huggingface.co/datasets/{CRUXEVAL_REPO}/resolve/{commit}/test.jsonl"
    with urlopen(source_url, timeout=60) as response:
        raw = response.read()
    converted = cruxeval_csv(raw)  # Validate before replacing local data.
    directory = data_dir / "raw/cruxeval"
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "cruxeval.csv"
    previous_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest() if csv_path.exists() else None
    for name, content in (("test.jsonl", raw), ("cruxeval.csv", converted)):
        temporary = directory / (name + ".download")
        temporary.write_bytes(content)
        temporary.replace(directory / name)
    record = {
        "path": "data/raw/cruxeval/cruxeval.csv",
        "source_path": "data/raw/cruxeval/test.jsonl",
        "dataset": CRUXEVAL_REPO,
        "requested_revision": revision,
        "revision": commit,
        "source_url": source_url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "sha256": hashlib.sha256(converted).hexdigest(),
        "bytes": len(converted),
        "record_count": len(load_cruxeval(csv_path)),
        "transformation": "JSONL to UTF-8 CSV; fields code,id,input,output; text values unchanged",
        "validation": "schema and unique IDs checked; reference correctness not checked",
        "previous_csv_sha256": previous_hash,
    }
    manifest = [entry for entry in manifest if entry.get("path") != record["path"]]
    manifest.insert(0, record)
    temporary = data_dir / "provenance.json.download"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return record
