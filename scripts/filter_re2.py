"""Exclude CRUXEval overlaps and the ten reviewed problematic RE2 samples."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_re2_export, write_re2_csv
from re2_filter import filter_re2


if __name__ == "__main__":
    rows = load_re2_export(ROOT / "data/raw/re2/re2_bench_lite_output_prediction_prompts.jsonl")
    kept, excluded = filter_re2(rows)
    # Keep the CSV exports beside the existing CRUXEval CSV.
    csv_dir = ROOT / "data/raw/cruxeval"
    write_re2_csv(rows, csv_dir / "re2_full.csv")
    write_re2_csv(kept, csv_dir / "re2_filtered.csv")
    output = ROOT / "data/processed/re2"
    output.mkdir(parents=True, exist_ok=True)
    (output / "re2_filtered.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    (output / "exclusions.json").write_text(json.dumps(excluded, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} samples -> {len(kept)} retained, {len(excluded)} excluded")
