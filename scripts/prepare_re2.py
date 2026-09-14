"""Write the screened 446-case RE2 selection and its inspection report."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavior_prediction.re2_selection import prepare_re2, write_re2_selection


if __name__ == "__main__":
    rows, report = prepare_re2(ROOT / "data")
    destination = ROOT / "data/processed/re2"
    write_re2_selection(rows, report, destination)
    print(f"500 inputs -> {len(rows)} retained; 44 overlaps + 10 problematic cases excluded; 7 inputs repaired.")
    print(destination)
