import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze import condition_rows, property_groups, thirds


class ConditionRowsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data = Path(self.temp.name)
        (self.data / "predictions").mkdir()
        self.tables = {"part_re2": [{"dataset": "re2", "id": "a", "repetition": 0}],
                       "part_crux": [{"dataset": "cruxeval", "id": "b", "repetition": 0}]}

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, run, prompt_file, effort="medium"):
        configuration = {"model": "m", "parameters": {"reasoning": {"effort": effort}}, "prompt_files": {prompt_file: "x"}}
        (self.data / "predictions" / f"{run}.manifest.json").write_text(json.dumps({"configuration": configuration}))

    def rows(self, runs):
        return condition_rows(runs, self.data, table=lambda run, data_dir: self.tables[run])

    def test_combines_runs_that_differ_only_in_datasets(self):
        self.manifest("part_re2", "re2_prompts.jsonl")
        self.manifest("part_crux", "cruxeval_prompts.jsonl")
        self.assertEqual([r["id"] for r in self.rows(["part_re2", "part_crux"])], ["a", "b"])

    def test_rejects_different_settings_and_repeated_items(self):
        self.manifest("part_re2", "re2_prompts.jsonl")
        self.manifest("part_crux", "cruxeval_prompts.jsonl", effort="low")
        with self.assertRaises(ValueError):
            self.rows(["part_re2", "part_crux"])
        self.manifest("part_crux", "cruxeval_prompts.jsonl")
        self.tables["part_crux"] = self.tables["part_re2"]
        with self.assertRaises(ValueError):
            self.rows(["part_re2", "part_crux"])


class SizeThirdsTests(unittest.TestCase):
    def test_thirds_partition_items_and_keep_ties_together(self):
        sizes = [1, 2, 2, 2, 3, 4, 5, 6, 6, 9]
        rows = [{"dataset": "cruxeval", "code_tokens": size, "executable_lines": None, "cyclomatic_complexity": None,
                 "target_cyclomatic_complexity": None, "halstead_difficulty": None, "target_halstead_difficulty": None,
                 "information_flow": None} for size in sizes]
        groups = [g for g in property_groups(rows) if g[2] == "code_tokens"]
        members = [[row["code_tokens"] for row in rows if member(row)] for *_, member in groups]
        self.assertEqual(sorted(sum(members, [])), sizes)
        self.assertTrue(all(members))
        for value in set(sizes):
            self.assertEqual(sum(value in third for third in members), 1, value)
        self.assertEqual([round(bound, 2) for bound in thirds(sizes)[0]], [1, 2.0])

    def test_skips_properties_without_enough_distinct_values(self):
        rows = [{"dataset": "re2", "code_tokens": 5, "executable_lines": 3, "cyclomatic_complexity": 2,
                 "target_cyclomatic_complexity": 1, "halstead_difficulty": 4.5, "target_halstead_difficulty": 2.0,
                 "information_flow": 4} for _ in range(6)]
        self.assertEqual(property_groups(rows), [])


if __name__ == "__main__":
    unittest.main()
