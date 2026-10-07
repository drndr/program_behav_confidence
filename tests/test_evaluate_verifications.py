import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_verifications import evaluate
from datasets import sha256_file


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def verdict_logprobs(word, probability):
    pieces = [("<verdict>", -0.01), (word, math.log(probability)), ("</verdict>", -0.01)]
    return {"content": [{"token": t, "bytes": list(t.encode()), "logprob": lp} for t, lp in pieces]}


class VerificationEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "data"
        prompts = self.data / "prompts/verification/pred/cruxeval_prompts.jsonl"
        write_jsonl(prompts, [{"dataset": "cruxeval", "id": f"s{i}", "prediction_attempt": 1} for i in range(4)])
        prompts.with_name("cruxeval_prompts.manifest.json").write_text(
            json.dumps({"prediction_run": {"path": "data/predictions/pred.jsonl"}}), encoding="utf-8")
        write_jsonl(self.data / "evaluations/pred_scores.jsonl",
                    [{"dataset": "cruxeval", "id": f"s{i}", "repetition": 0, "attempt": 1, "mode": mode,
                      "correct": i % 2 == 0} for i in range(4) for mode in ("strict", "relaxed", "normalized")])
        self.log = self.data / "verifications/verify.jsonl"
        write_jsonl(self.log, [
            {"dataset": "cruxeval", "id": "s0", "repetition": 0, "status": "completed",
             "text": "<verdict>correct</verdict><confidence>90</confidence>", "logprobs": verdict_logprobs("correct", 0.8)},
            {"dataset": "cruxeval", "id": "s1", "repetition": 0, "status": "completed",
             "text": "<verdict>incorrect</verdict><confidence>30</confidence>", "logprobs": None},
            {"dataset": "cruxeval", "id": "s2", "repetition": 0, "status": "error"},
            {"dataset": "cruxeval", "id": "s3", "repetition": 0, "status": "completed", "text": "<verdict>correct</verdict>"}])
        self.log.with_name("verify.manifest.json").write_text(json.dumps(
            {"configuration": {"prompt_files": {"cruxeval_prompts.jsonl": sha256_file(prompts)}}}), encoding="utf-8")

    def run_evaluation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return {r["id"]: r for r in evaluate(self.log, self.data)}

    def test_verdicts_joined_to_the_judged_answers(self):
        rows = self.run_evaluation()
        self.assertEqual([rows[i]["status"] for i in ("s0", "s1", "s2", "s3")],
                         ["judged", "judged", "request_error", "invalid_judgment"])
        self.assertEqual((rows["s0"]["verdict"], rows["s0"]["confidence"], rows["s0"]["correct_normalized"]), ("correct", 90.0, True))
        self.assertAlmostEqual(rows["s0"]["p_correct"], 0.8)
        self.assertEqual((rows["s1"]["verdict"], rows["s1"]["p_correct"], rows["s1"]["correct_strict"]), ("incorrect", None, False))
        summary = json.loads((self.data / "evaluations/verify_verdicts.summary.json").read_text())
        self.assertEqual(summary["prediction_run"], (self.data / "predictions/pred.jsonl").as_posix())
        self.assertEqual(summary["summary"]["cruxeval"]["verdict_matches_normalized"], 2)

    def test_refuses_an_evaluation_of_a_different_attempt(self):
        scores = self.data / "evaluations/pred_scores.jsonl"
        scores.write_text(scores.read_text().replace('"attempt": 1', '"attempt": 2'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "re-evaluate"):
            self.run_evaluation()


if __name__ == "__main__":
    unittest.main()
