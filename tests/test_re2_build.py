"""The tracked RE2 CSVs are exactly what scripts/filter_re2.py produces from the pinned release.

Runs only when the raw export and upstream zip are present (scripts/download_datasets.py re2).
Rows are compared field by field, so the row terminator style Git may apply on checkout
does not matter, but every code, input and output value must match byte for byte.
"""

import csv
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from filter_re2 import prepare_re2, remove_prompts  # also puts src on sys.path
from datasets import load_re2_export

COLUMNS = ["id", "code", "entrypoint", "input", "output"]


def tracked_rows(name):
    csv.field_size_limit(10_000_000)
    with (ROOT / "data/raw/re2" / name).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    return reader.fieldnames, rows


class Re2BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        provenance = ROOT / "data/provenance.json"
        source = None
        if provenance.exists():
            entries = json.loads(provenance.read_text(encoding="utf-8"))
            source = next((entry for entry in entries if "/re2/" in entry["path"]), None)
        if source is None or not all((ROOT / source[key]).exists() for key in ("path", "source_path")):
            raise unittest.SkipTest("RE2 raw export or upstream zip missing; run scripts/download_datasets.py re2")
        cls.source = source

    def assert_rows_match(self, name, produced):
        fieldnames, tracked = tracked_rows(name)
        self.assertEqual(fieldnames, COLUMNS, name)
        self.assertEqual(len(tracked), len(produced), f"{name}: row count")
        for index, (have, want) in enumerate(zip(tracked, produced)):
            for column in COLUMNS:
                self.assertEqual(have[column], want[column],
                                 f"{name} row {index} ({have['id']}): column {column} differs from script output")

    def test_filtered_csv_matches_script_output(self):
        final, report = prepare_re2(ROOT / "data")
        self.assertEqual((report["retained_count"], report["stdin_programs"]), (446, 34))
        self.assert_rows_match("re2_filtered.csv", final)

    def test_full_csv_matches_script_output(self):
        rows = remove_prompts(load_re2_export(ROOT / self.source["path"]))
        self.assertEqual(len(rows), 500)
        self.assert_rows_match("re2_full.csv", rows)


if __name__ == "__main__":
    unittest.main()
