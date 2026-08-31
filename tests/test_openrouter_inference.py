import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "inference" / "run_openrouter_inference.py"
SPEC = importlib.util.spec_from_file_location("run_openrouter_inference", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class OpenRouterInferenceTest(unittest.TestCase):
    def load_config(self, filename):
        config_path = (
            REPO_ROOT
            / "scripts"
            / "inference"
            / "configs"
            / filename
        )
        return runner.load_config(config_path)

    def test_cruxeval_gpt_none_config_is_pinned_and_complete(self):
        config = self.load_config(
            "openrouter_cruxeval_direct_gpt_5_4_nano_none.json"
        )

        self.assertEqual(config["model_name"], "openai/gpt-5.4-nano")
        self.assertEqual(
            config["model_revision"],
            "openai/gpt-5.4-nano-20260317",
        )
        self.assertEqual(
            config["reasoning"],
            {"effort": "none", "exclude": False},
        )
        self.assertEqual(config["generation"], {"max_tokens": 256, "seed": 0})
        self.assertEqual(config["expected_provider_name"], "OpenAI")
        self.assertEqual(config["provider"]["only"], ["openai"])
        self.assertFalse(config["provider"]["allow_fallbacks"])
        self.assertTrue(config["provider"]["require_parameters"])
        self.assertEqual(config["prompt_mode"], "direct")
        prompt_path = REPO_ROOT / config["prompt_path"]
        records = [
            json.loads(line)
            for line in prompt_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertEqual(len(records), 800)
        self.assertEqual({record["dataset"] for record in records}, {"cruxeval"})
        self.assertEqual(
            {record["prompt_version"] for record in records},
            {"cruxeval_raw_output_v1"},
        )

    def test_cruxeval_gpt_low_config_is_matched_reasoning_condition(self):
        none = self.load_config(
            "openrouter_cruxeval_direct_gpt_5_4_nano_none.json"
        )
        low = self.load_config(
            "openrouter_cruxeval_direct_gpt_5_4_nano_low.json"
        )

        self.assertEqual(
            {
                key
                for key in none.keys() | low.keys()
                if none.get(key) != low.get(key)
            },
            {"metadata_path", "output_path", "reasoning", "generation"},
        )
        self.assertEqual(low["reasoning"], {"effort": "low", "exclude": False})
        self.assertEqual(low["generation"], {"max_tokens": 1024, "seed": 0})
        self.assertEqual(low["model_name"], "openai/gpt-5.4-nano")
        self.assertEqual(
            low["model_revision"],
            "openai/gpt-5.4-nano-20260317",
        )
        self.assertEqual(low["expected_provider_name"], "OpenAI")
        self.assertEqual(low["provider"]["only"], ["openai"])
        self.assertFalse(low["provider"]["allow_fallbacks"])
        self.assertTrue(low["provider"]["require_parameters"])

    def test_request_body_keeps_reasoning_and_provider_controls(self):
        record = {"messages": [{"role": "user", "content": "Return 1."}]}
        config = {
            "model_name": "openai/gpt-5.4-nano",
            "reasoning": {"effort": "low", "exclude": False},
            "provider": {
                "only": ["openai"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
            "generation": {
                "max_tokens": 1024,
                "seed": 0,
            },
        }
        body = runner.build_request_body(record, config)

        self.assertEqual(body["model"], config["model_name"])
        self.assertEqual(body["messages"], record["messages"])
        self.assertEqual(body["reasoning"], config["reasoning"])
        self.assertEqual(body["provider"], config["provider"])
        self.assertEqual(body["max_tokens"], 1024)
        self.assertEqual(body["seed"], 0)
        self.assertFalse(body["stream"])
        self.assertEqual(body["usage"], {"include": True})

    def test_parse_completion_preserves_reasoning_and_usage(self):
        payload = {
            "id": "gen-1",
            "model": "openai/gpt-5.4-nano-20260317",
            "provider": "OpenAI",
            "created": 1,
            "openrouter_metadata": {
                "endpoints": {
                    "available": [
                        {
                            "provider": "OpenAI",
                            "model": "openai/gpt-5.4-nano-20260317",
                            "selected": True,
                        }
                    ]
                }
            },
            "choices": [
                {
                    "finish_reason": "stop",
                    "native_finish_reason": "STOP",
                    "message": {
                        "content": "42",
                        "reasoning": "Short thought summary.",
                        "reasoning_details": [
                            {
                                "type": "reasoning.summary",
                                "summary": "Short thought summary.",
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 12,
                "total_tokens": 112,
                "completion_tokens_details": {"reasoning_tokens": 8},
                "cost": 0.001,
            },
        }

        parsed = runner.parse_completion(payload)

        self.assertEqual(parsed["response"], "42")
        self.assertEqual(parsed["reasoning"], "Short thought summary.")
        self.assertEqual(parsed["finish_reason"], "stop")
        self.assertEqual(parsed["provider"], "OpenAI")
        self.assertEqual(runner.reasoning_token_count(parsed["usage"]), 8)

    def test_validate_resolved_route_rejects_provider_drift(self):
        completion = {
            "response_model": "openai/gpt-5.4-nano-20260317",
            "provider": "Other Provider",
            "openrouter_metadata": {
                "endpoints": {
                    "available": [
                        {
                            "provider": "Other Provider",
                            "model": "openai/gpt-5.4-nano-20260317",
                            "selected": True,
                        }
                    ]
                }
            },
        }
        config = {
            "model_name": "openai/gpt-5.4-nano",
            "model_revision": "openai/gpt-5.4-nano-20260317",
            "expected_provider_name": "OpenAI",
        }
        with self.assertRaisesRegex(runner.OpenRouterRequestError, "Resolved provider"):
            runner.validate_resolved_route(completion, config)

    def test_config_rejects_provider_parameter_dropping(self):
        config_path = (
            REPO_ROOT
            / "scripts"
            / "inference"
            / "configs"
            / "openrouter_cruxeval_direct_gpt_5_4_nano_low.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["provider"]["require_parameters"] = False
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory) / "invalid_openrouter_config.json"
            temporary_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "require_parameters"):
                runner.load_config(temporary_path)

    def test_incomplete_http_response_is_retryable(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.side_effect = runner.IncompleteRead(
            b'{"partial":'
        )
        with mock.patch.object(runner, "urlopen", return_value=response):
            with self.assertRaises(runner.OpenRouterRequestError) as raised:
                runner.post_json(
                    "https://openrouter.ai/api/v1/chat/completions",
                    "secret",
                    {"model": "test", "messages": []},
                    30.0,
                )
        self.assertTrue(raised.exception.retryable)
        self.assertIsNone(raised.exception.status_code)

    def test_connection_reset_is_retryable(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.side_effect = ConnectionResetError(
            "connection reset"
        )
        with mock.patch.object(runner, "urlopen", return_value=response):
            with self.assertRaises(runner.OpenRouterRequestError) as raised:
                runner.post_json(
                    "https://openrouter.ai/api/v1/chat/completions",
                    "secret",
                    {"model": "test", "messages": []},
                    30.0,
                )
        self.assertTrue(raised.exception.retryable)

    def test_read_existing_rejects_stale_or_malformed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps({"id": "sample", "config_hash": "stale"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                runner.read_existing(path, "current", {"sample"})
            path.write_text('{"id": "sample"', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Malformed checkpoint"):
                runner.read_existing(path, "current", {"sample"})

    def test_failed_request_aborts_without_poisoning_checkpoint(self):
        record = {
            "id": "sample-1",
            "dataset": "synthetic",
            "messages": [{"role": "user", "content": "Return 1."}],
        }
        config = {
            "model_name": "openai/gpt-5.4-nano",
            "model_revision": "openai/gpt-5.4-nano-20260317",
            "api_url": "https://openrouter.ai/api/v1/chat/completions",
            "expected_provider_name": "OpenAI",
            "reasoning": {"effort": "low", "exclude": False},
            "provider": {
                "only": ["openai"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
            "generation": {"max_tokens": 1024, "seed": 0},
            "retries": 0,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "predictions.jsonl"
            with mock.patch.object(
                runner,
                "post_json",
                side_effect=runner.OpenRouterRequestError(
                    "HTTP 401: invalid key",
                    retryable=False,
                    status_code=401,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "sample-1"):
                    runner.run_prompts(
                        [record],
                        "direct",
                        output_path,
                        config,
                        "config-hash",
                        "secret",
                    )
            self.assertEqual(output_path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
