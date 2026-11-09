from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from error_triage import checked, errors_caught


class ErrorsCaughtTests(unittest.TestCase):
    def test_lowest_scores_are_checked_first(self):
        scores, correct = [1, 2, 3, 4, 5], [False, True, True, False, True]
        self.assertEqual(errors_caught(scores, correct, 0.2), 0.5)
        self.assertEqual(errors_caught(scores, correct, 0.8), 1.0)
        self.assertEqual(errors_caught([5, 4, 3, 2, 1], correct, 0.2), 0.0)

    def test_ties_share_the_budget_like_random_checking(self):
        self.assertAlmostEqual(errors_caught([1] * 10, [False, True] * 5, 0.2), 0.2)
        # One sure error, then two of the four tied answers, one of which is an error.
        self.assertAlmostEqual(errors_caught([0, 1, 1, 1, 1], [False, False, True, True, True], 0.6), 0.75)

    def test_budget_is_rounded_down_to_whole_answers(self):
        self.assertEqual(checked(0.2, 9), 1)
        self.assertEqual(checked(0.2, 10), 2)
        scores, correct = list(range(9)), [False, False] + [True] * 7
        self.assertEqual(errors_caught(scores, correct, 0.2), 0.5)  # one answer checked, not 1.8

    def test_undefined_without_errors(self):
        self.assertIsNone(errors_caught([1, 2], [True, True], 0.5))


if __name__ == "__main__":
    unittest.main()
