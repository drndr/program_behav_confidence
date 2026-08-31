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
    "build_re2_full_calibration_prompts",
    REPO_ROOT
    / "scripts"
    / "prompt_generation"
    / "build_re2_full_calibration_prompts.py",
)


class Re2FullCalibrationPromptTest(unittest.TestCase):
    def test_final_response_keeps_output_contract_but_drops_rationale(self):
        response = (
            "[THOUGHT]\nsecret rationale\n[/THOUGHT]\n"
            "[OUTPUT]\n{\"output\": 3}\n[/OUTPUT]\ntrailing"
        )

        candidate = builder.candidate_final_response(response)

        self.assertEqual(candidate, '[OUTPUT]\n{"output": 3}\n[/OUTPUT]')
        self.assertNotIn("secret rationale", candidate)
        self.assertNotIn("trailing", candidate)

    def test_missing_output_marker_retains_response_for_format_audit(self):
        response = '```json\n{"output": 3}\n```'

        self.assertEqual(builder.candidate_final_response(response), response)

    def test_information_ablation_changes_only_program_access(self):
        candidate = '[OUTPUT]\n{"output": 3}\n[/OUTPUT]'
        original = "[PYTHON]\ndef f(): return 3\n[/PYTHON]"

        response_only = builder.build_response_only_prompt(candidate)
        code_aware = builder.build_code_aware_prompt(original, candidate)

        for prompt in (response_only, code_aware):
            self.assertIn(candidate, prompt)
            self.assertIn("integer percentage from 0 to 100", prompt)
            self.assertIn("safe compatibility", prompt)
        self.assertIn("intentionally hidden", response_only)
        self.assertNotIn(original, response_only)
        self.assertIn(original, code_aware)

    def test_all_confidence_configs_fix_low_judge_and_schema(self):
        root = REPO_ROOT / "scripts" / "inference" / "configs"
        for source in ("none", "low"):
            for information in ("response_only", "code_aware"):
                path = (
                    root
                    / f"openrouter_re2_full_confidence_{source}_{information}_"
                    "gpt_5_4_nano_low_judge.json"
                )
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(config["reasoning"]["effort"], "low")
                self.assertEqual(config["generation"]["seed"], 0)
                schema = config["generation"]["response_format"]["json_schema"]["schema"]
                self.assertEqual(schema["required"], ["confidence"])
                self.assertEqual(schema["properties"]["confidence"]["minimum"], 0)
                self.assertEqual(schema["properties"]["confidence"]["maximum"], 100)

    def test_base_configs_differ_only_by_reasoning_and_paths(self):
        root = REPO_ROOT / "scripts" / "inference" / "configs"
        configs = {
            condition: json.loads(
                (
                    root
                    / f"openrouter_re2_bench_full_gpt_5_4_nano_{condition}_"
                    "t4096_seed0.json"
                ).read_text(encoding="utf-8")
            )
            for condition in ("none", "low")
        }
        differing = {
            key
            for key in configs["none"].keys() | configs["low"].keys()
            if configs["none"].get(key) != configs["low"].get(key)
        }
        self.assertEqual(differing, {"reasoning", "output_path", "metadata_path"})
        self.assertEqual(configs["none"]["generation"], configs["low"]["generation"])


if __name__ == "__main__":
    unittest.main()
