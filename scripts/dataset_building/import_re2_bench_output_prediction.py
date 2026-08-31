"""Build local inference rows from a pinned external RE2-Bench checkout.

This adapter does not execute RE2-Bench code. It reads the released prompt
text, sample manifest, and serialized ground truth. Because the upstream
repository has no detected license file, outputs should remain in ``tmp/`` and
must not be redistributed until licensing is resolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ARTIFACT_URL = "https://github.com/AnonyRepo2025/RE2-Bench"
PINNED_COMMIT = "57b971f18567c879a7180bb6bd5b8e544dd9a47e"
DEFAULT_SEED = 20260806


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit-per-group", type=int)
    parser.add_argument("--expected-commit", default=PINNED_COMMIT)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def ground_truth_filename(problem_id: str, benchmark: str) -> str:
    if benchmark == "Swebench":
        parts = problem_id.split("@@")
        if len(parts) != 3:
            raise ValueError(f"Malformed Swebench problem ID: {problem_id}")
        # Preserve the released evaluator's filename convention exactly.
        return f"{parts[0]}@@{parts[1].rstrip('.py')}.{parts[2]}.jsonl"
    if benchmark == "cruxeval":
        return f"{problem_id.replace('sample', 'cruxeval')}.jsonl"
    if benchmark == "HumanEval":
        return f"{problem_id}.jsonl"
    if benchmark == "Classeval":
        return f"{problem_id.split('@')[-1].replace('.', '@')}.jsonl"
    if benchmark in {"Avatar", "Real"}:
        return f"{problem_id}.jsonl"
    raise ValueError(f"Unsupported RE2-Bench source: {benchmark}")


def read_jsonl_line(path: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError("JSONL index cannot be negative")
    with path.open(encoding="utf-8") as input_file:
        for line_index, line in enumerate(input_file):
            if line_index == index:
                return json.loads(line)
    raise IndexError(f"Index {index} is out of range for {path}")


def read_target(root: Path, problem_id: str, benchmark: str, io_index: int) -> Any:
    target_path = (
        root
        / "dataset"
        / "re2-bench"
        / "input-output"
        / benchmark
        / ground_truth_filename(problem_id, benchmark)
    )
    record = read_jsonl_line(target_path, io_index)
    values = record["values"]
    if "return" in values:
        return values["return"]
    if "output" in values:
        return values["output"]
    raise KeyError(f"No return or output value in {target_path}:{io_index + 1}")


def build_record(
    root: Path,
    problem_id: str,
    complexity_group: str,
    sample: dict[str, Any],
    artifact_commit: str,
) -> dict[str, Any]:
    benchmark = str(sample["benchmark"])
    io_index = int(sample["input-output"])
    case_name = ground_truth_filename(problem_id, benchmark)
    prompt_path = root / "prompts" / "output_prediction" / complexity_group / f"{problem_id}.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    target = read_target(root, problem_id, benchmark, io_index)
    return {
        "id": problem_id,
        "dataset": "re2_bench_lite",
        "task": "output_prediction",
        "source_dataset": benchmark,
        "complexity_group": complexity_group,
        "upstream_io_index": io_index,
        "upstream_case_name": case_name,
        "messages": [{"role": "user", "content": prompt}],
        # The local runner must retain the full response. RE2 tag parsing occurs later.
        "prompt_mode": "direct",
        "upstream_prompt_protocol": "icl_cot_with_output_structure_hint",
        "response_extraction": "re2_output_tag_json",
        "prompt_sha256": sha256_text(prompt),
        "target_output": json.dumps(target, ensure_ascii=False, sort_keys=True),
        "target_value": target,
        "target_serialization": "json",
        "artifact_url": ARTIFACT_URL,
        "artifact_commit": artifact_commit,
        "upstream_prompt_path": str(prompt_path.relative_to(root)).replace("\\", "/"),
    }


def interleave_groups(
    grouped_records: dict[str, list[dict[str, Any]]], seed: int
) -> list[dict[str, Any]]:
    if set(grouped_records) != {"HC", "LC"}:
        raise ValueError("Expected exactly HC and LC groups")
    hc = list(grouped_records["HC"])
    lc = list(grouped_records["LC"])
    if len(hc) != len(lc):
        raise ValueError(f"HC and LC must be balanced, found {len(hc)} and {len(lc)}")
    random.Random(seed).shuffle(hc)
    random.Random(seed + 1).shuffle(lc)
    rows = []
    start_with_hc = bool(seed % 2 == 0)
    for pair_index, (hc_row, lc_row) in enumerate(zip(hc, lc)):
        pair = (hc_row, lc_row) if start_with_hc else (lc_row, hc_row)
        for row in pair:
            rows.append({**row, "pair_index": pair_index, "order_index": len(rows)})
    return rows


def build_rows(
    root: Path,
    *,
    expected_commit: str = PINNED_COMMIT,
    seed: int = DEFAULT_SEED,
    limit_per_group: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = root.resolve()
    actual_commit = current_commit(root)
    if actual_commit is not None and actual_commit != expected_commit:
        raise RuntimeError(
            f"Artifact commit mismatch: expected {expected_commit}, found {actual_commit}"
        )
    manifest_path = root / "dataset" / "re2-bench" / "sampled_problems.json"
    with manifest_path.open(encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    if set(manifest) != {"HC", "LC"}:
        raise ValueError(f"Expected HC and LC in manifest, found {sorted(manifest)}")
    if limit_per_group is not None and limit_per_group <= 0:
        raise ValueError("limit_per_group must be positive")

    artifact_commit = actual_commit or expected_commit
    grouped_records: dict[str, list[dict[str, Any]]] = {}
    for group in ("HC", "LC"):
        items = list(manifest[group].items())
        if limit_per_group is not None:
            if limit_per_group > len(items):
                raise ValueError(
                    f"Requested {limit_per_group} {group} rows, only {len(items)} are available"
                )
            items = items[:limit_per_group]
        grouped_records[group] = [
            build_record(root, problem_id, group, sample, artifact_commit)
            for problem_id, sample in items
        ]
    rows = interleave_groups(grouped_records, seed)
    source_counts = Counter(
        (row["complexity_group"], row["source_dataset"]) for row in rows
    )
    metadata = {
        "dataset": "re2_bench_lite",
        "task": "output_prediction",
        "artifact_url": ARTIFACT_URL,
        "artifact_commit": artifact_commit,
        "artifact_root": str(root),
        "license_status": "unresolved_no_license_file_found",
        "redistribution_allowed": False,
        "seed": seed,
        "limit_per_group": limit_per_group,
        "row_count": len(rows),
        "group_counts": dict(Counter(row["complexity_group"] for row in rows)),
        "source_counts": {
            f"{group}:{source}": count
            for (group, source), count in sorted(source_counts.items())
        },
        "manifest_path": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "manifest_sha256": sha256_path(manifest_path),
        "response_extraction": "re2_output_tag_json",
    }
    return rows, metadata


def write_outputs(
    rows: list[dict[str, Any]], metadata: dict[str, Any], output: Path, metadata_output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    complete_metadata = {**metadata, "prompt_file_sha256": sha256_path(output)}
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(complete_metadata, output_file, indent=2, ensure_ascii=False, sort_keys=True)
        output_file.write("\n")


def main() -> None:
    args = parse_args()
    metadata_output = args.metadata_output or args.output.with_name(
        f"{args.output.stem}_metadata.json"
    )
    rows, metadata = build_rows(
        args.artifact_root,
        expected_commit=args.expected_commit,
        seed=args.seed,
        limit_per_group=args.limit_per_group,
    )
    write_outputs(rows, metadata, args.output, metadata_output)
    print(f"Rows:     {len(rows)}")
    print(f"Groups:   {metadata['group_counts']}")
    print(f"Prompts:  {args.output.resolve()}")
    print(f"Metadata: {metadata_output.resolve()}")


if __name__ == "__main__":
    main()
