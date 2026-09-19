import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_examples


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
        row = load_examples(self.path, "cruxeval")[0]
        self.assertEqual((row["dataset"], row["entrypoint"]), ("cruxeval", "f"))
        self.assertEqual((row["code"], row["input"], row["output"]), (code, "'a', 'b'", ""))

    def test_rejects_duplicate_ids(self):
        self.write_csv([["def f(): pass", "x", "", "None"]] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_examples(self.path, "cruxeval")

    def test_rejects_missing_csv_fields(self):
        self.write_csv([["def f(): pass", "x", ""]])
        with self.assertRaisesRegex(ValueError, "Malformed"):
            load_examples(self.path, "cruxeval")
        self.write_csv([["def f(): pass", "x", "", "None"]])
        with self.assertRaisesRegex(ValueError, "entrypoint"):
            load_examples(self.path, "re2")

    def test_re2_csv_preserves_stdin_and_reference_text(self):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "code", "entrypoint", "input", "output"])
            writer.writerow(["x", "print(input())", "", '"hello"', '"hello"'])
        row = load_examples(self.path, "re2")[0]
        self.assertEqual(row["entrypoint"], "")
        self.assertEqual(row["dataset"], "re2")
        self.assertEqual(row["input"], '"hello"')
        self.assertEqual(row["output"], '"hello"')

if __name__ == "__main__":
    unittest.main()
