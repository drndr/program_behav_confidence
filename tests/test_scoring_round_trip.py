"""The scorer must accept every reference of the tracked datasets and reject a wrong answer everywhere.

Feeds each reference back as a prediction (a perfect model's answer) and a sentinel
wrong answer, in every mode. Exact counts: a change in either data or scorer that
alters them must be made on purpose.
"""

from collections import Counter
import hashlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_examples
from scoring import SCORING_MODES, reference_value, score_prediction
from prompt_template import prediction_prompt

DATASETS = {"cruxeval": ("data/raw/cruxeval/cruxeval.csv", 800), "re2": ("data/raw/re2/re2_filtered.csv", 446)}


class RoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = {name: load_examples(ROOT / path, name) for name, (path, _) in DATASETS.items()}

    def test_every_reference_scores_correct_in_every_mode(self):
        for name, (_, count) in DATASETS.items():
            rows = self.examples[name]
            self.assertEqual(len(rows), count)
            for mode in SCORING_MODES:
                answers = (row["output"] if name == "cruxeval" else repr(reference_value(row)) for row in rows)
                outcomes = Counter((score_prediction(row, f"<prediction>{answer}</prediction>", mode)["status"],
                                    score_prediction(row, f"<prediction>{answer}</prediction>", mode)["correct"])
                                   for row, answer in zip(rows, answers))
                self.assertEqual(outcomes, {("scored", True): count}, (name, mode))

    def test_wrong_answer_is_wrong_everywhere(self):
        sentinel = "<prediction>'__no_such_answer__'</prediction>"
        for name, (_, count) in DATASETS.items():
            for mode in SCORING_MODES:
                outcomes = Counter(score_prediction(row, sentinel, mode)["correct"] for row in self.examples[name])
                self.assertEqual(outcomes, {False: count}, (name, mode))

    def test_prompts_render_and_are_distinct(self):
        hashes = [hashlib.sha256(prediction_prompt(row).encode("utf-8")).hexdigest()
                  for rows in self.examples.values() for row in rows]
        self.assertEqual(len(hashes), len(set(hashes)), "duplicate prompts would mean duplicated items")


if __name__ == "__main__":
    unittest.main()
