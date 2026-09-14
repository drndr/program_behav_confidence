"""Download RE2-Bench-lite sources and assemble its 500 output-prediction records."""

import argparse
import json
from pathlib import Path

from behavior_prediction.re2_download import REVISION, download_re2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=REVISION, help="Full upstream commit SHA")
    args = parser.parse_args()
    print(json.dumps(download_re2(Path(__file__).resolve().parents[1] / "data", args.revision), indent=2))


if __name__ == "__main__":
    main()
