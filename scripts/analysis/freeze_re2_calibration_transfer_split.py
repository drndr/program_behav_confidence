"""Freeze a deterministic RE2 calibration/test split for cross-benchmark transport."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_ROOT / "tmp/re2_bench_full_confidence_low_code_aware_v1.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "research_design/2026-08-19_re2_calibration_transfer_split.json"
SEED = 20260819


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        rows = [json.loads(line) for line in input_file if line.strip()]
    if len(rows) != 500:
        raise ValueError(f"Expected 500 RE2 prompt rows, found {len(rows)}")
    return rows


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_split(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    strata: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        stratum = "|".join(
            (
                str(row["complexity_group"]),
                str(row["source_dataset"]),
                f"correct_{int(bool(row['safe_compatibility_correct']))}",
                f"matched_{int(bool(row['matched_132_subset']))}",
            )
        )
        strata[stratum].append(str(row["id"]))

    target_calibration = len(rows) // 2
    calibration_counts = {stratum: len(ids) // 2 for stratum, ids in strata.items()}
    remaining = target_calibration - sum(calibration_counts.values())
    odd_strata = [stratum for stratum, ids in strata.items() if len(ids) % 2]
    odd_strata.sort(key=lambda value: sha256_text(f"{seed}|allocation|{value}"))
    for stratum in odd_strata[:remaining]:
        calibration_counts[stratum] += 1

    assignments: dict[str, str] = {}
    for stratum, ids in sorted(strata.items()):
        ordered = sorted(
            ids, key=lambda sample_id: sha256_text(f"{seed}|{stratum}|{sample_id}")
        )
        calibration_count = calibration_counts[stratum]
        for index, sample_id in enumerate(ordered):
            assignments[sample_id] = (
                "calibration" if index < calibration_count else "test"
            )
    counts = Counter(assignments.values())
    expected_counts = Counter(
        {"calibration": target_calibration, "test": len(rows) - target_calibration}
    )
    if counts != expected_counts:
        raise AssertionError(f"Unexpected split counts: {counts}")
    return assignments


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"Refusing to overwrite frozen split: {OUTPUT_PATH}")
    rows = read_jsonl(SOURCE_PATH)
    assignments = assign_split(rows, SEED)
    by_id = {str(row["id"]): row for row in rows}
    split_summary: dict[str, Any] = {}
    for split in ("calibration", "test"):
        selected = [by_id[sample_id] for sample_id, value in assignments.items() if value == split]
        split_summary[split] = {
            "n": len(selected),
            "correct": sum(bool(row["safe_compatibility_correct"]) for row in selected),
            "accuracy": sum(bool(row["safe_compatibility_correct"]) for row in selected)
            / len(selected),
            "complexity": dict(Counter(str(row["complexity_group"]) for row in selected)),
            "source_dataset": dict(Counter(str(row["source_dataset"]) for row in selected)),
            "matched_132": sum(bool(row["matched_132_subset"]) for row in selected),
        }
    payload = {
        "freeze": "re2_calibration_transfer_split_v1",
        "created_date": "2026-08-19",
        "seed": SEED,
        "source_path": str(SOURCE_PATH),
        "source_sha256": sha256_path(SOURCE_PATH),
        "confidence_values_read": False,
        "stratification": [
            "complexity_group",
            "source_dataset",
            "safe_compatibility_correct",
            "matched_132_subset",
        ],
        "allocation": (
            "Within-stratum SHA-256 ordering; floor(n/2) to calibration, with odd-stratum "
            "remainders assigned by a seeded SHA-256 allocation ordering to obtain 250/250."
        ),
        "summary": split_summary,
        "assignments": dict(sorted(assignments.items())),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(split_summary, indent=2, sort_keys=True))
    print(f"Frozen split: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
