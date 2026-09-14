"""Download the pinned CRUXEval and RE2-Bench-lite sources into data/raw.

Standard library only. Each download validates
structure, writes files atomically, and records provenance in data/provenance.json.
Reference outputs are stored unchanged; nothing here certifies their correctness.

    python scripts/download_datasets.py            # both datasets
    python scripts/download_datasets.py re2        # one dataset
"""

import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

CRUXEVAL_REPO = "cruxeval-org/cruxeval"
RE2_REPOSITORY = "https://github.com/AnonyRepo2025/RE2-Bench"
RE2_REVISION = "57b971f18567c879a7180bb6bd5b8e544dd9a47e"
RE2_ORDERING_SEED = 20260806


# --- shared helpers -----------------------------------------------------------

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path, content):
    temporary = path.with_name(path.name + ".download")
    temporary.write_bytes(content)
    temporary.replace(path)


def record_provenance(data_dir, record):
    """Replace the entry with the same path in place, or append a new one."""
    path = data_dir / "provenance.json"
    entries = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else []
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("Expected a list of provenance records")
    index = next((i for i, entry in enumerate(entries) if entry.get("path") == record["path"]), len(entries))
    entries[index:index + 1] = [record]
    write_atomic(path, (json.dumps(entries, indent=2) + "\n").encode("utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


# --- CRUXEval -----------------------------------------------------------------

def convert_cruxeval(raw):
    """Convert upstream JSONL to CSV without interpreting Python expressions.

    Returns (csv bytes, row count). The CSV is read back and compared with the
    JSONL rows so that no text value changes in the conversion.
    """
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
    converted = output.getvalue().encode("utf-8")
    with io.StringIO(converted.decode("utf-8"), newline="") as stream:
        if list(csv.DictReader(stream)) != rows:
            raise ValueError("CSV round trip changed CRUXEval values")
    return converted, len(rows)


def download_cruxeval(data_dir, revision="main"):
    """Resolve a revision, download immutable JSONL, and save a compatible CSV."""
    api_url = f"https://huggingface.co/api/datasets/{CRUXEVAL_REPO}/revision/{quote(revision, safe='')}"
    with urlopen(api_url, timeout=60) as response:
        commit = json.load(response)["sha"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Unexpected Hugging Face revision SHA")
    source_url = f"https://huggingface.co/datasets/{CRUXEVAL_REPO}/resolve/{commit}/test.jsonl"
    print(f"Downloading CRUXEval at {commit[:12]}...", flush=True)
    with urlopen(source_url, timeout=60) as response:
        raw = response.read()
    converted, count = convert_cruxeval(raw)  # Validate before replacing local data.
    directory = data_dir / "raw/cruxeval"
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "cruxeval.csv"
    previous = sha256_file(csv_path) if csv_path.exists() else None
    write_atomic(directory / "test.jsonl", raw)
    write_atomic(csv_path, converted)
    record = {
        "path": "data/raw/cruxeval/cruxeval.csv",
        "source_path": "data/raw/cruxeval/test.jsonl",
        "dataset": CRUXEVAL_REPO,
        "requested_revision": revision,
        "revision": commit,
        "source_url": source_url,
        "downloaded_at": now(),
        "source_sha256": sha256_bytes(raw),
        "sha256": sha256_bytes(converted),
        "bytes": len(converted),
        "record_count": count,
        "transformation": "JSONL to UTF-8 CSV; fields code,id,input,output; text values unchanged",
        "validation": "schema and unique IDs checked; reference correctness not checked",
        "previous_csv_sha256": previous,
    }
    record_provenance(data_dir, record)
    return record


# --- RE2-Bench-lite -----------------------------------------------------------

def re2_target_filename(problem_id, source):
    if source == "Swebench":
        project, filename, method = problem_id.split("@@")
        # Match the released evaluator's convention, including rstrip semantics.
        return f"{project}@@{filename.rstrip('.py')}.{method}.jsonl"
    if source == "cruxeval":
        return problem_id.replace("sample", "cruxeval") + ".jsonl"
    if source == "Classeval":
        return problem_id.split("@")[-1].replace(".", "@") + ".jsonl"
    if source in {"HumanEval", "Avatar", "Real"}:
        return problem_id + ".jsonl"
    raise ValueError(f"Unknown RE2 source: {source}")


def assemble_re2(archive_path, revision):
    """Return the 500 ordered output-prediction records and the release manifest bytes."""
    groups = {}
    with zipfile.ZipFile(archive_path) as archive:
        prefix = f"RE2-Bench-{revision}/"
        manifest_bytes = archive.read(prefix + "dataset/re2-bench/sampled_problems.json")
        manifest = json.loads(manifest_bytes)
        if set(manifest) != {"HC", "LC"} or any(len(manifest[group]) != 250 for group in manifest):
            raise ValueError("Expected RE2-Bench-lite: 250 HC and 250 LC problems")
        for group in ("HC", "LC"):
            rows = []
            for problem_id, sample in manifest[group].items():
                source = sample["benchmark"]
                io_index = sample["input-output"]
                if not isinstance(io_index, int) or io_index < 0:
                    raise ValueError(f"Invalid input/output index: {problem_id}")
                case = re2_target_filename(problem_id, source)
                prompt_path = f"prompts/output_prediction/{group}/{problem_id}.txt"
                # Match text-mode newline handling in the historical importer.
                prompt = archive.read(prefix + prompt_path).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
                target_path = f"dataset/re2-bench/input-output/{source}/{case}"
                lines = archive.read(prefix + target_path).decode("utf-8").splitlines()
                values = json.loads(lines[io_index])["values"]
                if "return" not in values and "output" not in values:
                    raise ValueError(f"Missing target: {problem_id}")
                target = values["return"] if "return" in values else values["output"]
                rows.append({
                    "id": problem_id, "dataset": "re2_bench_lite", "task": "output_prediction",
                    "source_dataset": source, "complexity_group": group,
                    "upstream_io_index": io_index, "upstream_case_name": case,
                    "messages": [{"role": "user", "content": prompt}],
                    "prompt_mode": "direct", "upstream_prompt_protocol": "icl_cot_with_output_structure_hint",
                    "response_extraction": "re2_output_tag_json",
                    "prompt_sha256": sha256_bytes(prompt.encode()),
                    "target_output": json.dumps(target, ensure_ascii=False, sort_keys=True),
                    "target_value": target, "target_serialization": "json",
                    "artifact_url": RE2_REPOSITORY, "artifact_commit": revision,
                    "upstream_prompt_path": prompt_path,
                })
            random.Random(RE2_ORDERING_SEED if group == "HC" else RE2_ORDERING_SEED + 1).shuffle(rows)
            groups[group] = rows
    rows = []
    for pair_index, pair in enumerate(zip(groups["HC"], groups["LC"])):
        for row in pair:
            rows.append({**row, "pair_index": pair_index, "order_index": len(rows)})
    if len({row["id"] for row in rows}) != 500:
        raise ValueError("Duplicate RE2 identifiers")
    return rows, manifest_bytes


def serialize_re2(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def download_re2(data_dir, revision=RE2_REVISION):
    """Download the pinned upstream zip and assemble the 500-record export."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full upstream commit SHA")
    directory = data_dir / "raw/re2"
    directory.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/AnonyRepo2025/RE2-Bench/zip/{revision}"
    archive_path = directory / f"upstream-{revision}.zip"
    temporary = directory / "upstream.zip.download"
    print(f"Downloading RE2-Bench at {revision[:12]}...", flush=True)
    with urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    rows, manifest_bytes = assemble_re2(temporary, revision)  # Validate before replacing local data.
    output = directory / "re2_bench_lite_output_prediction_prompts.jsonl"
    previous = sha256_file(output) if output.exists() else None
    temporary.replace(archive_path)
    write_atomic(output, serialize_re2(rows))
    (directory / "sampled_problems.json").write_bytes(manifest_bytes)
    record = {
        "path": "data/raw/re2/" + output.name,
        "source_path": "data/raw/re2/" + archive_path.name,
        "source_url": url, "artifact_url": RE2_REPOSITORY, "revision": revision,
        "downloaded_at": now(),
        "sha256": sha256_file(output), "source_sha256": sha256_file(archive_path),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "record_count": len(rows), "bytes": output.stat().st_size,
        "group_counts": dict(Counter(row["complexity_group"] for row in rows)),
        "source_counts": dict(Counter(row["source_dataset"] for row in rows)),
        "ordering_seed": RE2_ORDERING_SEED, "previous_export_sha256": previous,
        "validation": "release manifest and unique IDs checked; targets unchanged and not repaired or certified",
    }
    record_provenance(data_dir, record)
    return record


# --- command line -------------------------------------------------------------

DOWNLOADS = {"cruxeval": download_cruxeval, "re2": download_re2}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", metavar="DATASET",
                        help=f"which to download: {', '.join(DOWNLOADS)} (default: all)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="target data directory")
    parser.add_argument("--cruxeval-revision", default="main", help="Hugging Face revision or commit SHA")
    parser.add_argument("--re2-revision", default=RE2_REVISION, help="full upstream commit SHA")
    args = parser.parse_args()
    selected = args.datasets or list(DOWNLOADS)
    unknown = [name for name in selected if name not in DOWNLOADS]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}; choose from {', '.join(DOWNLOADS)}")
    revisions = {"cruxeval": args.cruxeval_revision, "re2": args.re2_revision}
    for name in dict.fromkeys(selected):
        print(json.dumps(DOWNLOADS[name](args.data_dir, revisions[name]), indent=2))


if __name__ == "__main__":
    main()
