import importlib.util
import json
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


builder = load_module(
    "build_cruxeval_calibration_pilot",
    REPO_ROOT / "scripts" / "prompt_generation" / "build_cruxeval_calibration_pilot.py",
)
analysis = load_module(
    "analyze_cruxeval_calibration_pilot",
    REPO_ROOT / "scripts" / "analysis" / "analyze_cruxeval_calibration_pilot.py",
)


class CalibrationPromptBuilderTest(unittest.TestCase):
    def test_paired_split_is_exact_and_preserves_all_ids(self):
        sample_ids = {f"sample_{index}" for index in range(10)}
        none_correct = [True, True, True, True, False, False, False, False, False, True]
        low_correct = [True, True, False, False, True, True, False, False, True, True]
        scores = {
            "none": {
                sample_id: {"relaxed_correct": none_correct[index]}
                for index, sample_id in enumerate(sorted(sample_ids))
            },
            "low": {
                sample_id: {"relaxed_correct": low_correct[index]}
                for index, sample_id in enumerate(sorted(sample_ids))
            },
        }

        assignments, strata = builder.assign_paired_splits(sample_ids, scores, seed=7)

        self.assertEqual(set(assignments), sample_ids)
        self.assertEqual(sum(split == "calibration" for split in assignments.values()), 5)
        self.assertEqual(sum(split == "test" for split in assignments.values()), 5)
        self.assertEqual(sum(strata.values()), 10)
        repeated, _ = builder.assign_paired_splits(sample_ids, scores, seed=7)
        self.assertEqual(assignments, repeated)

    def test_prompt_explicitly_requests_integer_percentage_and_semantic_value(self):
        prompt = builder.build_prompt("def f():\n    return 1", "None", "1")

        self.assertIn("integer percentage from 0 to 100", prompt)
        self.assertIn("Judge the value", prompt)
        self.assertIn('<proposed_output>\n1\n</proposed_output>', prompt)


class CalibrationAnalysisTest(unittest.TestCase):
    def test_confidence_parser_accepts_schema_and_rejects_fractional_scale(self):
        self.assertEqual(analysis.parse_confidence('{"confidence": 73}'), 0.73)
        self.assertEqual(analysis.parse_confidence('```json\n{"confidence": 100}\n```'), 1.0)
        with self.assertRaises(ValueError):
            analysis.parse_confidence('{"confidence": 101}')

    def test_confidence_parser_reports_empty_truncated_output(self):
        with self.assertRaisesRegex(ValueError, "Could not parse confidence"):
            analysis.parse_confidence("")

    def test_auc_handles_ties(self):
        y = np.asarray([0, 0, 1, 1], dtype=int)
        probability = np.asarray([0.1, 0.4, 0.4, 0.9], dtype=float)

        self.assertAlmostEqual(analysis.roc_auc(y, probability), 0.875)

    def test_platt_scaling_improves_systematic_overconfidence(self):
        probability = np.asarray(
            [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99],
            dtype=float,
        )
        y = np.asarray([0, 0, 0, 0, 0, 1, 0, 1, 1, 1], dtype=int)

        model = analysis.fit_platt(probability, y)
        calibrated = analysis.apply_platt(probability, model)

        self.assertLess(analysis.brier_score(y, calibrated), analysis.brier_score(y, probability))

    def test_threshold_selection_maximizes_qualifying_coverage(self):
        y = np.asarray([1, 1, 1, 0, 1, 0], dtype=int)
        probability = np.asarray([0.99, 0.95, 0.90, 0.80, 0.70, 0.60], dtype=float)

        selected = analysis.select_accuracy_threshold(
            y, probability, target_accuracy=0.75, minimum_items=2
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["accepted"], 5)
        self.assertAlmostEqual(selected["threshold"], 0.70)
        self.assertAlmostEqual(selected["accuracy"], 0.8)

    def test_risk_coverage_does_not_split_confidence_ties(self):
        y = np.asarray([1, 0, 1, 0], dtype=int)
        probability = np.asarray([1.0, 1.0, 0.5, 0.1], dtype=float)

        first = analysis.risk_coverage(y, probability)[0]

        self.assertEqual(first["requested_coverage"], 0.25)
        self.assertEqual(first["accepted"], 2)
        self.assertEqual(first["actual_coverage"], 0.5)


class CalibrationConfigTest(unittest.TestCase):
    def test_configs_are_paired_and_request_structured_confidence(self):
        config_root = REPO_ROOT / "scripts" / "inference" / "configs"
        configs = {}
        for condition in ("none", "low"):
            path = (
                config_root
                / f"openrouter_cruxeval_reflective_confidence_gpt_5_4_nano_{condition}.json"
            )
            configs[condition] = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            {
                key
                for key in configs["none"].keys() | configs["low"].keys()
                if configs["none"].get(key) != configs["low"].get(key)
            },
            {"prompt_path", "output_path", "metadata_path", "reasoning", "generation"},
        )
        for condition, config in configs.items():
            self.assertEqual(config["reasoning"]["effort"], condition)
            schema = config["generation"]["response_format"]["json_schema"]["schema"]
            self.assertEqual(schema["required"], ["confidence"])
            self.assertEqual(schema["properties"]["confidence"]["minimum"], 0)
            self.assertEqual(schema["properties"]["confidence"]["maximum"], 100)
            self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
