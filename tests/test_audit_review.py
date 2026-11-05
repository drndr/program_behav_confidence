from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_agreement import kappa, review_verdicts
from make_audit_review import stratified_sample, verdict


def audit_rows(counts):
    rows = []
    for (solvable, category), count in counts.items():
        rows += [{"id": f"{category}_{i}", "solvable": solvable, "category": category} for i in range(count)]
    return rows


class StratifiedSampleTests(unittest.TestCase):
    def test_strata_are_proportional_with_largest_remainders(self):
        rows = audit_rows({("False", "input_lost"): 84, ("False", "reference_wrong"): 25,
                           ("False", "environment"): 15, ("True", "missed_computation"): 27})
        sample = stratified_sample(rows, 40)
        by_id = {row["id"]: verdict(row) for row in rows}
        self.assertEqual(Counter(by_id[i] for i in sample),
                         {"input_lost": 22, "reference_wrong": 7, "environment": 4, "solvable": 7})

    def test_the_seed_fixes_the_sample(self):
        rows = audit_rows({("False", "input_lost"): 30, ("True", "missed_library"): 10})
        self.assertEqual(stratified_sample(rows, 8, seed=1), stratified_sample(rows, 8, seed=1))
        self.assertEqual(len(set(stratified_sample(rows, 8, seed=1))), 8)


class KappaTests(unittest.TestCase):
    def test_kappa(self):
        self.assertEqual(kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]), 1.0)
        self.assertAlmostEqual(kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]), 0.5)
        self.assertIsNone(kappa(["a", "a"], ["a", "a"]))  # chance agreement is certain


class ReviewVerdictsTests(unittest.TestCase):
    def test_reads_labelled_items_of_an_export(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.csv"
            path.write_text("id,rater,verdict,reason\na,Ann,solvable,long loop\nb,Ann,input_lost,no state\nc,Ann,,\n",
                            encoding="utf-8")
            self.assertEqual(review_verdicts(path), ("Ann", {"a": ("solvable", "long loop"), "b": ("input_lost", "no state")}))


if __name__ == "__main__":
    unittest.main()
