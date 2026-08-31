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


analysis = load_module(
    "analyze_cruxeval_calibration_deconfounders",
    REPO_ROOT
    / "scripts"
    / "analysis"
    / "analyze_cruxeval_calibration_deconfounders.py",
)


class CalibrationDeconfounderTest(unittest.TestCase):
    def test_intercept_update_preserves_fixed_slope(self):
        probability = np.asarray([0.1, 0.2, 0.7, 0.8, 0.9], dtype=float)
        y = np.asarray([0, 0, 1, 1, 1], dtype=int)

        fitted = analysis.fit_calibration_model(probability, y, fixed_slope=0.5)

        self.assertAlmostEqual(fitted["slope"], 0.5)
        self.assertTrue(fitted["success"])
        self.assertTrue(np.isfinite(fitted["intercept"]))

    def test_full_calibration_fit_improves_reversed_confidence(self):
        probability = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=float)
        y = np.asarray([0, 0, 1, 1], dtype=int)

        fitted = analysis.fit_calibration_model(probability, y)
        calibrated = analysis.apply_calibration(probability, fitted)

        self.assertLess(fitted["slope"], 0)
        self.assertLess(
            analysis.mean_log_loss(y, calibrated),
            analysis.mean_log_loss(y, probability),
        )

    def test_shared_type_counts_require_both_outcomes_in_both_conditions(self):
        rows = {
            "none": [
                {"target_type": "int", "semantic_correct": True},
                {"target_type": "int", "semantic_correct": False},
                {"target_type": "str", "semantic_correct": True},
            ],
            "low": [
                {"target_type": "int", "semantic_correct": True},
                {"target_type": "int", "semantic_correct": False},
                {"target_type": "str", "semantic_correct": True},
                {"target_type": "str", "semantic_correct": False},
            ],
        }

        counts = analysis.shared_type_counts(rows)

        self.assertEqual(counts["int"], 1)
        self.assertEqual(counts["str"], 0)

    def test_pipeline_bootstrap_refits_calibrators(self):
        source_probability = np.asarray([0.1, 0.2, 0.8, 0.9] * 5)
        source_y = np.asarray([0, 0, 1, 1] * 5)
        target_probability = np.asarray([0.2, 0.3, 0.7, 0.8] * 5)
        target_y = np.asarray([0, 0, 1, 1] * 5)

        result = analysis.bootstrap_transfer_pipeline(
            source_y,
            source_probability,
            target_y,
            target_probability,
            target_y,
            target_probability,
            repetitions=20,
            seed=7,
        )

        self.assertIn("source calibration", result["resampling"])
        self.assertEqual(len(result["brier_difference_95_ci"]), 2)
        self.assertEqual(len(result["target_minus_source_slope_95_ci"]), 2)

    def test_fixed_coverage_does_not_split_ties(self):
        items = {
            "none": [
                {
                    "id": str(index),
                    "split": "test",
                    "semantic_correct": index != 1,
                    "confidence": confidence,
                }
                for index, confidence in enumerate([1.0, 1.0, 0.8, 0.7, 0.1])
            ]
        }

        result = analysis.fixed_coverage_selective_risk(items, "none")

        first = result["fixed_coverage"][0]
        self.assertEqual(first["requested_coverage"], 0.70)
        self.assertEqual(first["accepted"], 4)
        self.assertAlmostEqual(first["actual_coverage"], 0.8)


if __name__ == "__main__":
    unittest.main()
