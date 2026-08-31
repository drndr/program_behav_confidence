import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = load_module(
    "audit_cruxeval_calibration_internal_validity",
    REPO_ROOT
    / "scripts"
    / "analysis"
    / "audit_cruxeval_calibration_internal_validity.py",
)


class CalibrationInternalValidityTest(unittest.TestCase):
    def test_stratified_auc_uses_only_within_group_pairs(self):
        rows = [
            {"group": "a", "semantic_correct": False, "confidence": 0.1},
            {"group": "a", "semantic_correct": True, "confidence": 0.9},
            {"group": "b", "semantic_correct": False, "confidence": 0.8},
            {"group": "b", "semantic_correct": True, "confidence": 0.2},
            {"group": "c", "semantic_correct": True, "confidence": 1.0},
        ]

        result = audit.stratified_auc(rows, lambda row: row["group"])

        self.assertEqual(result["comparable_pairs"], 2)
        self.assertAlmostEqual(result["auroc"], 0.5)
        self.assertEqual(len(result["strata_excluded"]), 1)

    def test_adjusted_confidence_recovers_signal_beyond_type_and_length(self):
        rows = []
        for index in range(80):
            correct = index % 2 == 0
            rows.append(
                {
                    "split": "test",
                    "semantic_correct": correct,
                    "confidence": 0.9 if correct else 0.1,
                    "target_type": "int" if index % 4 < 2 else "str",
                    "code_chars": 100 + index % 5,
                    "target_chars": 5 + index % 3,
                }
            )

        result = audit.adjusted_confidence_audit(rows)

        self.assertGreater(result["confidence_odds_ratio"], 1.0)
        self.assertLess(result["likelihood_ratio_p_value"], 0.001)
        self.assertTrue(result["full_fit_success"])

    def test_scoring_policy_audit_counts_changed_labels(self):
        rows = [
            {
                "split": "calibration",
                "semantic_correct": True,
                "strict_correct": False,
                "confidence": 0.8,
            },
            {
                "split": "calibration",
                "semantic_correct": False,
                "strict_correct": False,
                "confidence": 0.2,
            },
            {
                "split": "test",
                "semantic_correct": True,
                "strict_correct": False,
                "confidence": 0.8,
            },
            {
                "split": "test",
                "semantic_correct": False,
                "strict_correct": False,
                "confidence": 0.2,
            },
        ]

        result = audit.scoring_policy_audit(rows)

        self.assertEqual(result["label_changes"]["total_changed"], 1)
        self.assertAlmostEqual(result["semantic_value"]["accuracy"], 0.5)
        self.assertAlmostEqual(result["strict_serialization"]["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
