import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "analysis"))

from re2_bench_scoring import (
    parse_re2_response,
    preparsed_result,
    safe_compatibility_normalize,
    score_re2_output,
)
from score_re2_bench_predictions import score_rows, summarize_rows


class Re2BenchScoringTest(unittest.TestCase):
    def test_strict_tagged_json_is_protocol_correct(self):
        parsed = parse_re2_response('[THOUGHT]x[/THOUGHT]\n[OUTPUT]{"output": [1, 2]}[/OUTPUT]')
        score = score_re2_output(parsed, [1, 2])
        self.assertTrue(parsed.parsed)
        self.assertEqual(parsed.mode, "strict_json")
        self.assertTrue(score.strict_json_typed_correct)

    def test_missing_tag_is_not_silently_accepted(self):
        parsed = parse_re2_response('{"output": 1}')
        score = score_re2_output(parsed, 1)
        self.assertEqual(parsed.status, "missing_output_tag")
        self.assertFalse(score.safe_compatibility_correct)

    def test_repairs_unquoted_symbolic_value_without_eval(self):
        parsed = parse_re2_response('[OUTPUT]{"output": {Q.integer, Q.positive}}[/OUTPUT]')
        score = score_re2_output(parsed, "{Q.integer, Q.positive}")
        self.assertEqual(parsed.mode, "repaired_json")
        self.assertEqual(parsed.repair_count, 1)
        self.assertTrue(score.repaired_json_typed_correct)
        self.assertFalse(score.strict_json_typed_correct)

    def test_fenced_json_and_trailing_comma_are_safe(self):
        parsed = parse_re2_response('[OUTPUT]```json\n{"output": true,}\n```[/OUTPUT]')
        self.assertTrue(score_re2_output(parsed, True).strict_json_typed_correct)

    def test_safe_compatibility_handles_serialized_set_and_array(self):
        self.assertEqual(safe_compatibility_normalize("set()"), set())
        parsed = preparsed_result({"output": "[1 2 3]"})
        score = score_re2_output(parsed, "[1.0 2.0 3.0]")
        self.assertFalse(score.repaired_json_typed_correct)
        self.assertTrue(score.safe_compatibility_correct)

    def test_type_identity_rejects_bool_as_integer(self):
        parsed = parse_re2_response('[OUTPUT]{"output": true}[/OUTPUT]')
        score = score_re2_output(parsed, 1)
        self.assertFalse(score.safe_compatibility_correct)

    def test_compatibility_policy_ignores_outer_string_whitespace(self):
        parsed = preparsed_result({"output": "value\n"})
        score = score_re2_output(parsed, "value")
        self.assertFalse(score.repaired_json_typed_correct)
        self.assertTrue(score.safe_compatibility_correct)

    def test_truncation_metadata_is_preserved(self):
        parsed = parse_re2_response('[OUTPUT]{"output": 1}')
        score = score_re2_output(parsed, 1, finish_reason="length")
        self.assertTrue(score.is_truncated)
        self.assertTrue(score.strict_json_typed_correct)

    def test_summary_includes_source_matched_cells_and_intervals(self):
        rows = []
        for group, source, correct in (
            ("HC", "Real", True),
            ("LC", "Real", False),
            ("HC", "Swebench", False),
            ("LC", "Swebench", True),
        ):
            rows.append(
                {
                    "complexity_group": group,
                    "source_dataset": source,
                    "parse_status": "parsed",
                    "parse_mode": "strict_json",
                    "strict_json_typed_correct": correct,
                    "repaired_json_typed_correct": correct,
                    "safe_compatibility_correct": correct,
                    "is_truncated": False,
                }
            )
        summary = summarize_rows(rows)
        self.assertEqual(set(summary["by_complexity_and_source"]), {
            "HC:Real",
            "HC:Swebench",
            "LC:Real",
            "LC:Swebench",
        })
        policy = summary["by_complexity_and_source"]["HC:Real"]["policies"]["safe_compatibility_correct"]
        self.assertEqual(policy["system_accuracy"], 1.0)
        self.assertEqual(len(policy["wilson_95"]), 2)

    def test_factorial_treatment_metadata_survives_scoring(self):
        prompts = {
            "sample": {
                "id": "sample",
                "source_dataset": "Real",
                "complexity_group": "HC",
                "prompt_sha256": "prompt-hash",
                "target_value": 1,
            }
        }
        prediction = {
            "id": "sample",
            "model": "openai/gpt-5.4-nano",
            "model_revision": "openai/gpt-5.4-nano-20260317",
            "response_model": "openai/gpt-5.4-nano",
            "provider": "OpenAI",
            "config_hash": "config-hash",
            "arm_id": "high_t8192_seed1",
            "replicate_seed": 1,
            "reasoning_effort": "high",
            "max_tokens": 8192,
            "response": '[OUTPUT]{"output": 1}[/OUTPUT]',
            "finish_reason": "stop",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.jsonl"
            path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            row = score_rows(prompts, path)[0]
        self.assertEqual(row["arm_id"], "high_t8192_seed1")
        self.assertEqual(row["reasoning_effort"], "high")
        self.assertEqual(row["max_tokens"], 8192)
        self.assertEqual(row["replicate_seed"], 1)
        self.assertEqual(row["provider"], "OpenAI")
        self.assertEqual(row["config_hash"], "config-hash")


if __name__ == "__main__":
    unittest.main()
