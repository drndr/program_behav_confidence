"""Print a dataset inventory; run from any directory after installing the package."""

import argparse
import json
from collections import Counter
from pathlib import Path

from behavior_prediction.datasets import load_cruxeval, load_re2_export, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    crux_path = args.data_dir / "raw/cruxeval/cruxeval.csv"
    re2_path = args.data_dir / "raw/re2/re2_bench_lite_output_prediction_prompts.jsonl"
    crux = load_cruxeval(crux_path)
    re2 = load_re2_export(re2_path)
    print(json.dumps({
        "cruxeval": {"rows": len(crux), "sha256": sha256_file(crux_path), "status": "loaded; scoring pending"},
        "re2_bench_lite": {
            "rows": len(re2), "sha256": sha256_file(re2_path),
            "source_counts": dict(Counter(row.get("source_dataset", "unknown") for row in re2)),
            "status": "inspection only; eligibility and label validation pending",
        },
    }, indent=2))


if __name__ == "__main__":
    main()
