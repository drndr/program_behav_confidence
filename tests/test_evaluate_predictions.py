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
from datasets import load_examples


def write_run(directory, rows):
    path = Path(directory) / "pilot.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    manifest = {"configuration": {}}
    path.with_name("pilot.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


class EvaluationTests(unittest.TestCase):
    def test_summary_logprob_weights_answers_equally_and_excludes_missing(self):
        examples = load_examples(ROOT / "data/raw/cruxeval/cruxeval.csv", "cruxeval")[:3]
        rows = []
        for example, text, content in zip(examples, ("1", "[1, 2]", "3"),
                ([{"token": "1", "logprob": 0.0}],
                 [{"token": "[1", "logprob": -2.0}, {"token": ", 2]", "logprob": -4.0}], None)):
            logprobs = {"content": [{"token": "<prediction>", "logprob": -9.0},
                                   *content, {"token": "</prediction>", "logprob": -9.0}]} if content else None
            rows.append({"dataset": "cruxeval", "id": example["id"], "repetition": 0,
                         "status": "completed", "text": f"<prediction>{text}</prediction>",
                         "logprobs": logprobs})
        with tempfile.TemporaryDirectory() as directory:
            path = write_run(directory, rows)
            out = Path(directory) / "scores"
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate(path, out)
            summary = json.loads((out / "pilot_scores.summary.json").read_text())["summary"]
            for group in summary.values():
                self.assertEqual(group["with_logprobs"], 2)
                # Equal answer weights: (0 + (-2 - 4) / 2) / 2 = -1.5, not -2.
                self.assertEqual(group["mean_prediction_logprob"], -1.5)
            for row in rows:
                row["logprobs"] = None
            path = write_run(directory, rows)
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate(path, out)
            summary = json.loads((out / "pilot_scores.summary.json").read_text())["summary"]
            for group in summary.values():
                self.assertEqual(group["with_logprobs"], 0)
                self.assertIsNone(group["mean_prediction_logprob"])

    def test_latest_attempt_and_noncompleted_results(self):
        examples = load_examples(ROOT / "data/raw/cruxeval/cruxeval.csv", "cruxeval")[:4]
        rows = [{"dataset": "cruxeval", "id": e["id"], "repetition": 0, "attempt": 1, "status": "error"} for e in examples]
        rows[2]["status"] = "incomplete"
        rows[3].update(status="completed", text="no tags at all")
        answer = f"<prediction>{examples[0]['output']}</prediction>"
        logprobs = {"content": [{"token": answer, "bytes": list(answer.encode("utf-8")), "logprob": -0.25}]}
        rows.append(dict(rows[0], attempt=2, status="completed", text=answer, logprobs=logprobs, finish_reason="stop",
                         backend_provider="Together", usage={"completion_tokens": 40, "completion_tokens_details": {"reasoning_tokens": 30}}))
        with tempfile.TemporaryDirectory() as directory:
            path = write_run(directory, rows)
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
            self.assertEqual({(r["completion_tokens"], r["reasoning_tokens"], r["finish_reason"], r["backend_provider"])
                              for r in scored}, {(40, 30, "stop", "Together")})
            self.assertEqual(summary["summary"]["cruxeval/strict"]["with_logprobs"], 1)
            self.assertTrue(all(r["prediction_mean_logprob"] is None for r in results if r["status"] != "scored"))


if __name__ == "__main__":
    unittest.main()
