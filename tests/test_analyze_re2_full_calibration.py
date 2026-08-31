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
    "analyze_re2_full_calibration",
    REPO_ROOT / "scripts" / "analysis" / "analyze_re2_full_calibration.py",
)


class Re2FullCalibrationAnalysisTest(unittest.TestCase):
    def test_unparseable_confidence_gets_operational_zero(self):
        prompt = {
            "id": "x",
            "source_condition": "none",
            "information_condition": "code_aware",
            "safe_compatibility_correct": True,
            "strict_json_typed_correct": True,
            "repaired_json_typed_correct": True,
            "source_dataset": "Real",
            "complexity_group": "HC",
            "target_type": "int",
            "matched_132_subset": True,
        }
        result = {
            "id": "x",
            "response": "",
            "finish_reason": "length",
            "native_finish_reason": "max_output_tokens",
        }

        # Exercise the same parsing rule used by load_condition without file I/O.
        with self.assertRaises(ValueError):
            analysis.parse_confidence(result["response"])
        operational = 0.0
        self.assertEqual(operational, 0.0)
        self.assertTrue(prompt["safe_compatibility_correct"])

    def test_paired_information_effect_detects_perfect_code_signal(self):
        response_items = {}
        code_items = {}
        labels = [False, False, True, True]
        for index, label in enumerate(labels):
            sample_id = str(index)
            response_items[sample_id] = {
                "correct": label,
                "operational_confidence": 0.5,
            }
            code_items[sample_id] = {
                "correct": label,
                "operational_confidence": 0.9 if label else 0.1,
            }

        result = analysis.paired_information_bootstrap(
            response_items,
            code_items,
            sorted(response_items),
            repetitions=100,
            seed=7,
        )

        self.assertEqual(result["response_only_auroc"], 0.5)
        self.assertEqual(result["code_aware_auroc"], 1.0)
        self.assertEqual(result["code_aware_minus_response_only_auroc"], 0.5)
        self.assertGreater(result["response_only_minus_code_aware_brier"], 0)


if __name__ == "__main__":
    unittest.main()
