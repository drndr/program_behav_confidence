import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_predictions import parser, read_results, run


class FakeClient:
    provider = "openrouter"

    def __init__(self, fail_first=False):
        self.calls = 0
        self.fail_first = fail_first

    def complete(self, messages, model, parameters):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            return {"status": "error", "http_status": 500, "error": "test"}
        return {"status": "completed", "text": "<prediction>4</prediction>", "raw_response": {"test": True}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        prompt = {"id": "a", "dataset": "cruxeval", "messages": [{"role": "user", "content": "code"}],
                  "prompt_sha256": hashlib.sha256(b"code").hexdigest()}
        (self.root / "cruxeval_prompts.jsonl").write_text(json.dumps(prompt) + "\n", encoding="utf-8")
        self.args = parser().parse_args(["--model", "org/model", "--run-name", "test", "--datasets", "cruxeval",
                                       "--prompt-dir", str(self.root), "--out-dir", str(self.root), "--repetitions", "2"])

    def quiet_run(self, client):
        with contextlib.redirect_stdout(io.StringIO()): return run(self.args, client)

    def test_resume_skips_completed_requests(self):
        client = FakeClient()
        self.quiet_run(client)
        self.quiet_run(client)
        self.assertEqual(client.calls, 2)
        results = read_results(self.root / "test.jsonl")
        self.assertEqual([r["repetition"] for r in results], [0, 1])

    def test_configuration_mismatch_prevents_calls(self):
        client = FakeClient()
        self.quiet_run(client)
        self.args.model = "different/model"
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.quiet_run(client)
        self.assertEqual(client.calls, 2)
        self.assertFalse((self.root / "test.lock").exists())

    def test_error_retry_is_explicit_and_append_only(self):
        client = FakeClient(fail_first=True)
        self.quiet_run(client)
        self.quiet_run(client)
        self.assertEqual(client.calls, 2)
        self.args.retry_errors = True
        self.quiet_run(client)
        results = read_results(self.root / "test.jsonl")
        self.assertEqual(len(results), 3)
        self.assertEqual((results[-1]["repetition"], results[-1]["attempt"]), (0, 2))

    def test_torn_final_write_recovered_without_losing_previous_records(self):
        client = FakeClient()
        self.quiet_run(client)
        path = self.root / "test.jsonl"
        with path.open("ab") as f: f.write(b'{"partial":')
        self.quiet_run(client)
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(read_results(path)), 2)
        self.assertTrue(path.with_name("test.jsonl.partial").exists())

    def test_dry_run_makes_no_calls_or_run_files(self):
        self.args.dry_run = True
        client = FakeClient()
        self.assertEqual(self.quiet_run(client), {"planned": 2})
        self.assertEqual(client.calls, 0)
        self.assertFalse((self.root / "test.manifest.json").exists())

    def test_changed_prompt_rejected_before_call(self):
        path = self.root / "cruxeval_prompts.jsonl"
        row = json.loads(path.read_text())
        row["messages"][0]["content"] = "changed"
        path.write_text(json.dumps(row) + "\n")
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.quiet_run(client)
        self.assertEqual(client.calls, 0)


if __name__ == "__main__":
    unittest.main()
