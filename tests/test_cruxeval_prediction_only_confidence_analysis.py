import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analysis = load_module(
    "analyze_cruxeval_prediction_only_confidence_control",
    REPO_ROOT
    / "scripts"
    / "analysis"
    / "analyze_cruxeval_prediction_only_confidence_control.py",
)


def synthetic_rows():
    rows = []
    for split in ("calibration", "test"):
        for index in range(80):
            correct = index % 2 == 0
            rows.append(
                {
                    "id": f"{split}_{index}",
                    "split": split,
                    "semantic_correct": correct,
                    "code_aware_confidence": 0.9 if correct else 0.1,
                    "prediction_only_confidence": 0.5,
                    "target_type": "int" if index % 4 < 2 else "str",
                    "code_chars": 100 + index % 7,
                    "target_chars": 5 + index % 3,
                }
            )
    return rows


class PredictionOnlyAnalysisTest(unittest.TestCase):
    def test_point_comparison_favors_informative_code_aware_confidence(self):
        result = analysis.comparison_point_estimates(synthetic_rows())

        self.assertGreater(result["auroc_code_aware_minus_prediction_only"], 0.4)
        self.assertGreater(result["brier_prediction_only_minus_code_aware"], 0.0)

    def test_paired_bootstrap_keeps_information_conditions_aligned(self):
        result = analysis.paired_bootstrap(synthetic_rows(), repetitions=100, seed=7)

        interval = result["auroc_code_aware_minus_prediction_only"]
        self.assertIsNotNone(interval)
        self.assertGreater(interval[0], 0.4)

    def test_incremental_test_finds_code_aware_signal(self):
        result = analysis.incremental_confidence_audit(synthetic_rows())

        code_aware = result["code_aware_added_after_prediction_only"]
        self.assertLess(code_aware["likelihood_ratio_p_value"], 0.001)


if __name__ == "__main__":
    unittest.main()
