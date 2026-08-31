import importlib.util
import json
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


builder = load_module(
    "build_cruxeval_prediction_only_confidence_control",
    REPO_ROOT
    / "scripts"
    / "prompt_generation"
    / "build_cruxeval_prediction_only_confidence_control.py",
)


class PredictionOnlyPromptTest(unittest.TestCase):
    def test_prompt_contains_candidate_but_no_program_or_input_tags(self):
        prompt = builder.build_prompt("[1, 2]")

        self.assertIn("<proposed_output>\n[1, 2]\n</proposed_output>", prompt)
        self.assertIn("program and input are intentionally hidden", prompt)
        self.assertNotIn("<program>", prompt)
        self.assertNotIn("<input>", prompt)
        self.assertIn("integer percentage from 0 to 100", prompt)

    def test_configs_differ_only_in_paired_condition_fields(self):
        config_root = REPO_ROOT / "scripts" / "inference" / "configs"
        configs = {
            condition: json.loads(
                (
                    config_root
                    / f"openrouter_cruxeval_prediction_only_confidence_gpt_5_4_nano_{condition}.json"
                ).read_text(encoding="utf-8")
            )
            for condition in ("none", "low")
        }

        differing = {
            key
            for key in configs["none"].keys() | configs["low"].keys()
            if configs["none"].get(key) != configs["low"].get(key)
        }
        self.assertEqual(
            differing,
            {"prompt_path", "output_path", "metadata_path", "reasoning", "generation"},
        )
        for condition, config in configs.items():
            self.assertEqual(config["reasoning"]["effort"], condition)
            schema = config["generation"]["response_format"]["json_schema"]["schema"]
            self.assertEqual(schema["required"], ["confidence"])
            self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
