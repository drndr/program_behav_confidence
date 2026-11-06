from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paired_differences import PROPERTY_PAIRS, SIGNAL_PAIRS, SIZE, difference, paired_differences


def row(correct, size, complexity=None, reasoning=None, confidence=None, verdict=None):
    return {"dataset": "re2", "source": "Real", "correct_normalized": correct, "code_tokens": size,
            "cyclomatic_complexity": complexity, "reasoning_tokens": reasoning, "confidence": confidence,
            "verdict": verdict, "logprob": None}


class DifferenceTests(unittest.TestCase):
    def test_difference_of_auroc_within_strata(self):
        correct, strata = [True, False, True, False], ["a", "a", "b", "b"]
        self.assertEqual(difference([2, 1, 2, 1], [1, 2, 1, 2], correct, strata), 1.0)
        self.assertEqual(difference([2, 1, 2, 1], [2, 1, 2, 1], correct, strata), 0.0)
        self.assertIsNone(difference([1, 2], [2, 1], [True, True], ["a", "a"]))


class PairedDifferencesTests(unittest.TestCase):
    rows = [row(index % 2 == 0, 100 + index, complexity=20 - (index % 2) * 10 + index % 3, reasoning=index % 2,
                confidence=90, verdict="correct") for index in range(40)]

    def pair(self, records, first):
        return next(record for record in records if record["signal"] == first)

    def test_identical_quantities_differ_by_nothing_in_every_resample(self):
        same = [dict(r, cyclomatic_complexity=r["code_tokens"]) for r in self.rows]
        record = paired_differences(same, ("complexity (baseline)", SIZE), PROPERTY_PAIRS[:1], resamples=20)[0]
        self.assertEqual((record["difference"], record["low"], record["high"]), (0.0, 0.0, 0.0))
        self.assertFalse(record["excludes_zero"])

    def test_a_clear_gain_excludes_zero_and_answers_are_counted(self):
        record = paired_differences(self.rows, ("reasoning tokens", SIZE), SIGNAL_PAIRS[:1], resamples=50)[0]
        self.assertEqual(record["signal"], "reasoning tokens")
        self.assertEqual((record["answers"], record["wrong"]), (40, 20))
        self.assertGreater(record["low"], 0)
        self.assertTrue(record["excludes_zero"])

    def test_only_answers_that_have_every_shared_quantity_are_compared(self):
        rows = self.rows + [row(False, 500, reasoning=None), row(None, 600, reasoning=1)]
        record = paired_differences(rows, ("reasoning tokens", SIZE), SIGNAL_PAIRS[:1], resamples=5)[0]
        self.assertEqual(record["answers"], 40)

    def test_a_pair_is_skipped_where_a_quantity_is_missing_or_constant(self):
        self.assertEqual(paired_differences(self.rows, ("answer logprob", SIZE), SIGNAL_PAIRS[2:3], resamples=5), [])
        constant = [dict(r, reasoning_tokens=0) for r in self.rows]
        self.assertEqual(paired_differences(constant, ("reasoning tokens", SIZE), SIGNAL_PAIRS[:1], resamples=5), [])


if __name__ == "__main__":
    unittest.main()
