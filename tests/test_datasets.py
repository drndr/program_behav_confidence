import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_cruxeval, load_re2_export


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "fixture"

    def write_csv(self, rows):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["code", "id", "input", "output"])
            writer.writerows(rows)

    def test_preserves_multiline_code_arguments_and_empty_target(self):
        code = "def f(a, b):\n    return a + b\n"
        self.write_csv([[code, "x", "'a', 'b'", "" ]])
        row = load_cruxeval(self.path)[0]
        self.assertEqual((row.code, row.input, row.output), (code, "'a', 'b'", ""))

    def test_rejects_duplicate_ids(self):
        self.write_csv([["def f(): pass", "x", "", "None"]] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_cruxeval(self.path)

    def test_rejects_missing_field(self):
        self.write_csv([["def f(): pass", "x", ""]])
        with self.assertRaisesRegex(ValueError, "Malformed"):
            load_cruxeval(self.path)

    def test_preserves_re2_serialization_and_metadata(self):
        row = {"id": "x", "messages": [{"role": "user", "content": "original prompt"}],
               "target_value": "[1, 2]", "target_output": json.dumps("[1, 2]"),
               "source_dataset": "example", "unknown_metadata": {"keep": True}}
        self.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.assertEqual(load_re2_export(self.path), [row])

    def test_rejects_invalid_re2_messages(self):
        self.path.write_text(json.dumps({"id": "x", "messages": "prompt"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "messages"):
            load_re2_export(self.path)


if __name__ == "__main__":
    unittest.main()
