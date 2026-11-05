from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import auroc, bootstrap_interval, calibration, selective_accuracy, stratified_auroc


def brute_force_auroc(scores, correct):
    pos = [s for s, c in zip(scores, correct) if c]
    neg = [s for s, c in zip(scores, correct) if not c]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


class AurocTests(unittest.TestCase):
    def test_matches_brute_force_with_ties(self):
        rng = random.Random(1)
        for _ in range(50):
            scores = [rng.choice([0, 0.5, 1, 1.5, 2, 99, 100]) for _ in range(40)]
            correct = [rng.random() < 0.6 for _ in range(40)]
            if 0 < sum(correct) < 40:
                self.assertAlmostEqual(auroc(scores, correct), brute_force_auroc(scores, correct))

    def test_extremes_and_undefined(self):
        self.assertEqual(auroc([3, 2, 1], [True, True, False]), 1.0)
        self.assertEqual(auroc([1, 2, 3], [True, True, False]), 0.0)
        self.assertEqual(auroc([5, 5, 5], [True, False, False]), 0.5)
        self.assertIsNone(auroc([1, 2], [True, True]))
        with self.assertRaises(ValueError):
            auroc([1, 2], [True])
        with self.assertRaises(ValueError):
            auroc([1, None], [True, False])
        with self.assertRaises(ValueError):
            auroc([1, float("nan")], [True, False])


class StratifiedAurocTests(unittest.TestCase):
    def test_pairs_only_within_strata_match_brute_force(self):
        rng = random.Random(3)
        for _ in range(30):
            scores = [rng.choice([0, 1, 2, 3, 5]) for _ in range(60)]
            correct = [rng.random() < 0.5 for _ in range(60)]
            strata = [rng.randrange(3) for _ in range(60)]
            pairs = [(p, n) for i, p in enumerate(scores) for j, n in enumerate(scores)
                     if correct[i] and not correct[j] and strata[i] == strata[j]]
            expected = sum((p > n) + 0.5 * (p == n) for p, n in pairs) / len(pairs)
            self.assertAlmostEqual(stratified_auroc(scores, correct, strata), expected)

    def test_a_score_that_only_tells_strata_apart_is_chance(self):
        # Larger programs are wrong more often; a score equal to the stratum separates overall, not within.
        strata = [0, 0, 0, 0, 1, 1, 1, 1]
        correct = [True, True, True, False, True, False, False, False]
        scores = [-s for s in strata]
        self.assertGreater(auroc(scores, correct), 0.5)
        self.assertEqual(stratified_auroc(scores, correct, strata), 0.5)

    def test_one_stratum_is_plain_auroc_and_no_mixed_stratum_is_undefined(self):
        self.assertEqual(stratified_auroc([3, 2, 1], [True, True, False], ["a"] * 3), auroc([3, 2, 1], [True, True, False]))
        self.assertIsNone(stratified_auroc([1, 2], [True, False], ["a", "b"]))
        with self.assertRaises(ValueError):
            stratified_auroc([1, 2], [True, False], ["a"])


class BootstrapTests(unittest.TestCase):
    def test_reproducible_and_brackets_the_estimate(self):
        rng = random.Random(2)
        scores = [rng.random() for _ in range(200)]
        correct = [rng.random() < s for s in scores]
        first = bootstrap_interval(auroc, scores, correct, resamples=300)
        self.assertEqual(first, bootstrap_interval(auroc, scores, correct, resamples=300))
        self.assertLess(first[0], auroc(scores, correct))
        self.assertGreater(first[1], auroc(scores, correct))
        self.assertNotEqual(first, bootstrap_interval(auroc, scores, correct, resamples=300, seed=1))

    def test_undefined_statistic(self):
        self.assertIsNone(bootstrap_interval(auroc, [1, 2, 3], [True, True, True], resamples=50))


class SelectiveAccuracyTests(unittest.TestCase):
    def test_keeps_the_most_confident(self):
        scores, correct = [0.9, 0.8, 0.2, 0.1], [True, True, False, True]
        self.assertEqual(selective_accuracy(scores, correct, 0.5), 1.0)
        self.assertEqual(selective_accuracy(scores, correct, 1.0), 0.75)

    def test_ties_at_the_cut_count_in_proportion(self):
        # One answer at 100 (right), then four tied at 99 (two right): keeping three takes 2 of the 4.
        scores, correct = [100, 99, 99, 99, 99], [True, True, False, True, False]
        self.assertAlmostEqual(selective_accuracy(scores, correct, 0.6), (1 + 2 * 0.5) / 3)

    def test_rejects_bad_coverage(self):
        with self.assertRaises(ValueError):
            selective_accuracy([1], [True], 0)


class CalibrationTests(unittest.TestCase):
    def test_perfect_and_overconfident(self):
        self.assertEqual(calibration([0.5, 0.5], [True, False])["ece"], 0.0)
        result = calibration([1.0, 1.0, 1.0, 1.0], [True, False, False, False])
        self.assertAlmostEqual(result["ece"], 0.75)
        self.assertEqual([(b["low"], b["count"]) for b in result["bins"]], [(0.9, 4)])

    def test_rejects_values_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            calibration([99], [True])


if __name__ == "__main__":
    unittest.main()
