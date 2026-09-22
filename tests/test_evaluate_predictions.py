import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_predictions import evaluate
from datasets import sha256_file
from datasets import load_examples


def write_run(directory, rows, prompt_files):
    path = Path(directory) / "pilot.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    manifest = {"configuration": {"prompt_files": prompt_files}}
    path.with_name("pilot.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.prompt_files = {"cruxeval_prompts.jsonl": sha256_file(ROOT / "data/prompts/cruxeval_prompts.jsonl")}

    def test_latest_attempt_and_noncompleted_results(self):
        examples = load_examples(ROOT / "data/raw/cruxeval/cruxeval.csv", "cruxeval")[:4]
        rows = [{"dataset": "cruxeval", "id": e["id"], "repetition": 0, "attempt": 1, "status": "error"} for e in examples]
        rows[2]["status"] = "incomplete"
        rows[3].update(status="completed", text="no tags at all")
        answer = f"<prediction>{examples[0]['output']}</prediction>"
        logprobs = {"content": [{"token": answer, "bytes": list(answer.encode("utf-8")), "logprob": -0.25}]}
        rows.append(dict(rows[0], attempt=2, status="completed", text=answer, logprobs=logprobs))
        with tempfile.TemporaryDirectory() as directory:
            path = write_run(directory, rows, self.prompt_files)
            with contextlib.redirect_stdout(io.StringIO()):
                results = evaluate(path, Path(directory) / "scores")
            self.assertEqual(len(results), 12)
            summary = json.loads((Path(directory) / "scores/pilot_scores.summary.json").read_text())
            for mode in ("strict", "relaxed", "normalized"):
                group = summary["summary"][f"cruxeval/{mode}"]
                # Only requests without an answer are unscored; a malformed answer is wrong.
                self.assertEqual((group["total"], group["correct"], group["incorrect"], group["unscored"]), (4, 1, 1, 2))
                self.assertEqual(group["statuses"], {"scored": 1, "invalid_response": 1,
                                                     "request_error": 1, "request_incomplete": 1})
            self.assertTrue(all("\\" not in key for key in summary["sha256"]))

            scored = [r for r in results if r["status"] == "scored"]
            self.assertEqual({r["prediction_mean_logprob"] for r in scored}, {-0.25})
            self.assertEqual(summary["summary"]["cruxeval/strict"]["with_logprobs"], 1)
            self.assertTrue(all(r["prediction_mean_logprob"] is None for r in results if r["status"] != "scored"))

    def test_refuses_run_whose_prompts_changed(self):
        rows = [{"dataset": "cruxeval", "id": "sample_0", "repetition": 0, "status": "error"}]
        with tempfile.TemporaryDirectory() as directory:
            path = write_run(directory, rows, {"cruxeval_prompts.jsonl": "0" * 64})
            with self.assertRaisesRegex(ValueError, "differs from the prompts"):
                evaluate(path, Path(directory) / "scores")
            self.assertFalse((Path(directory) / "scores").exists())


if __name__ == "__main__":
    unittest.main()
