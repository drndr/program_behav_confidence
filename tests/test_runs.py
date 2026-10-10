import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from runs import item_table, verification_of


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class ItemTableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "data"
        scores = []
        for item_id, correct, status in (("sample_0", True, "scored"), ("sample_1", None, "request_incomplete")):
            for mode in ("strict", "relaxed", "normalized"):
                scores.append({"dataset": "cruxeval", "id": item_id, "repetition": 0, "attempt": 1, "mode": mode,
                               "correct": correct, "status": status, "prediction_mean_logprob": -0.1 if correct else None,
                               "completion_tokens": 50, "reasoning_tokens": 40, "finish_reason": "stop", "backend_provider": "X"})
        write_jsonl(self.data / "evaluations/pred_scores.jsonl", scores)
        properties = self.data / "program_properties/cruxeval_properties.csv"
        properties.parent.mkdir(parents=True)
        with properties.open("w", encoding="utf-8", newline="") as stream:
            halstead = ["halstead_volume", "halstead_difficulty", "target_halstead_volume", "target_halstead_difficulty",
                        "information_flow"]
            writer = csv.DictWriter(stream, fieldnames=["id", "code_token_length", "input_token_length",
                                                        "executable_lines", "line_status", "cyclomatic_complexity",
                                                        "target_cyclomatic_complexity", *halstead])
            writer.writeheader()
            writer.writerows([{"id": "sample_0", "code_token_length": 30, "input_token_length": 5, "executable_lines": 4,
                               "line_status": "ok", "cyclomatic_complexity": 3, "target_cyclomatic_complexity": 2,
                               **dict(zip(halstead, (100.5, 7.25, 80.0, 6.0, 144)))},
                              {"id": "sample_1", "code_token_length": 60, "input_token_length": 9, "executable_lines": "",
                               "line_status": "tokenization_failed", "cyclomatic_complexity": "",
                               "target_cyclomatic_complexity": "", **dict.fromkeys(halstead, "")}])

    def add_verification(self, name):
        (self.data / f"evaluations/{name}_verdicts.summary.json").write_text(
            json.dumps({"prediction_run": "data/predictions/pred.jsonl"}), encoding="utf-8")
        write_jsonl(self.data / f"evaluations/{name}_verdicts.jsonl",
                    [{"dataset": "cruxeval", "id": "sample_0", "repetition": 0, "status": "judged", "verdict": "correct",
                      "confidence": 95.0, "p_correct": 0.9, "reasoning_tokens": 12}])

    def test_joins_scores_properties_and_verdicts(self):
        self.assertEqual({r["id"]: r["verification_run"] for r in item_table("pred", self.data)},
                         {"sample_0": None, "sample_1": None})
        self.add_verification("verify-pred")
        rows = {r["id"]: r for r in item_table("pred", self.data)}
        first, second = rows["sample_0"], rows["sample_1"]
        self.assertEqual((first["correct_normalized"], first["logprob"], first["reasoning_tokens"], first["code_tokens"]),
                         (True, -0.1, 40, 30))
        self.assertEqual((first["source"], first["verdict"], first["confidence"], first["p_correct"]),
                         ("CRUXEval", "correct", 95.0, 0.9))
        self.assertEqual((second["correct_normalized"], second["status_normalized"], second["executable_lines"], second["verdict"]),
                         (None, "request_incomplete", None, None))
        self.assertEqual((first["cyclomatic_complexity"], first["target_cyclomatic_complexity"]), (3, 2))
        self.assertEqual((second["cyclomatic_complexity"], second["target_cyclomatic_complexity"]), (None, None))
        self.assertEqual((first["halstead_difficulty"], first["target_halstead_volume"], second["halstead_volume"]),
                         (7.25, 80.0, None))
        self.assertEqual((first["information_flow"], second["information_flow"]), (144, None))

    def test_at_most_one_verification_per_run(self):
        self.add_verification("verify-a")
        self.add_verification("verify-b")
        with self.assertRaises(ValueError):
            verification_of("pred", self.data)


if __name__ == "__main__":
    unittest.main()
