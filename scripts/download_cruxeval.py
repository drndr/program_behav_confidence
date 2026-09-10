"""Download CRUXEval from Hugging Face into data/raw/cruxeval."""

import argparse
import json
from pathlib import Path

from behavior_prediction.downloads import download_cruxeval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Hugging Face revision or commit SHA")
    args = parser.parse_args()
    record = download_cruxeval(Path(__file__).resolve().parents[1] / "data", args.revision)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
