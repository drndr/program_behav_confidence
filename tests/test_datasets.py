import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_examples, load_re2_export, re2_source

RE2_EXPORT = ROOT / "data/raw/re2/re2_bench_lite_output_prediction_prompts.jsonl"


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

    def test_re2_source_from_id(self):
        cases = {"HumanEval_44": "HumanEval", "ClassEval_24@ComplexCalculator.multiply": "Classeval",
                 "atcoder_ABC132_A": "Avatar", "codeforces_1_A": "Avatar",
                 "django__django-11728@@django.contrib.admindocs.utils.py@@replace_unnamed_groups": "Swebench",
                 "sympy@@sympy_core_exprtools.py@@_mask_nc_L1272": "Real", "sample_0": None}
        for item_id, source in cases.items():
            self.assertEqual(re2_source(item_id), source, item_id)

    @unittest.skipUnless(RE2_EXPORT.exists(), "local RE2 export not downloaded")
    def test_re2_source_matches_the_export_for_every_selected_item(self):
        export = {row["id"]: row["source_dataset"] for row in load_re2_export(RE2_EXPORT)}
        for example in load_examples(ROOT / "data/raw/re2/re2_filtered.csv", "re2"):
            self.assertEqual(re2_source(example["id"]), export[example["id"]], example["id"])


if __name__ == "__main__":
    unittest.main()
