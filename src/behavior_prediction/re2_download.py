"""Download the pinned RE2-Bench-lite output-prediction release without execution."""

import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from behavior_prediction.datasets import sha256_file


REPOSITORY = "https://github.com/AnonyRepo2025/RE2-Bench"
REVISION = "57b971f18567c879a7180bb6bd5b8e544dd9a47e"


def target_filename(problem_id: str, source: str) -> str:
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


def download_re2(data_dir: str | Path, revision: str = REVISION) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full upstream commit SHA")
    data_dir = Path(data_dir)
    provenance_path = data_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig")) if provenance_path.exists() else []
    directory = data_dir / "raw/re2"
    directory.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/AnonyRepo2025/RE2-Bench/zip/{revision}"
    archive_path = directory / f"upstream-{revision}.zip"
    temporary = directory / "upstream.zip.download"
    print("Downloading pinned RE2 release...", flush=True)
    with urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    groups = {}
    seed = 20260806
    with zipfile.ZipFile(temporary) as archive:
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
                case = target_filename(problem_id, source)
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
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "target_output": json.dumps(target, ensure_ascii=False, sort_keys=True),
                    "target_value": target, "target_serialization": "json",
                    "artifact_url": REPOSITORY, "artifact_commit": revision,
                    "upstream_prompt_path": prompt_path,
                })
            random.Random(seed if group == "HC" else seed + 1).shuffle(rows)
            groups[group] = rows
    rows = []
    for pair_index, pair in enumerate(zip(groups["HC"], groups["LC"])):
        for row in pair:
            rows.append({**row, "pair_index": pair_index, "order_index": len(rows)})
    if len({row["id"] for row in rows}) != 500:
        raise ValueError("Duplicate RE2 identifiers")
    output = directory / "re2_bench_lite_output_prediction_prompts.jsonl"
    previous = sha256_file(output) if output.exists() else None
    output_temp = directory / (output.name + ".download")
    with output_temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(archive_path)
    output_temp.replace(output)
    (directory / "sampled_problems.json").write_bytes(manifest_bytes)
    record = {
        "path": "data/raw/re2/" + output.name,
        "source_path": "data/raw/re2/" + archive_path.name,
        "source_url": url, "artifact_url": REPOSITORY, "revision": revision,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256_file(output), "source_sha256": sha256_file(archive_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "record_count": len(rows), "bytes": output.stat().st_size,
        "group_counts": dict(Counter(row["complexity_group"] for row in rows)),
        "source_counts": dict(Counter(row["source_dataset"] for row in rows)),
        "ordering_seed": seed, "previous_export_sha256": previous,
        "validation": "release manifest and unique IDs checked; targets unchanged and not repaired or certified",
    }
    provenance = [entry for entry in provenance if entry.get("path") != record["path"]]
    provenance.append(record)
    manifest_temp = data_dir / "provenance.json.download"
    manifest_temp.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    manifest_temp.replace(provenance_path)
    return record
