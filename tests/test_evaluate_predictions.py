import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_predictions import evaluate
from predictions import load_examples


class EvaluationTests(unittest.TestCase):
    def test_latest_attempt_and_noncompleted_results(self):
        root = Path(__file__).resolve().parents[1]
        examples = load_examples(root / "data/raw/cruxeval/cruxeval.csv", "cruxeval")[:3]
        rows = [{"dataset": "cruxeval", "id": e["id"], "repetition": 0,
                 "attempt": 1, "status": "error"} for e in examples]
        rows[2]["status"] = "incomplete"
        rows.append(dict(rows[0], attempt=2, status="completed",
                         text=f"<prediction>{examples[0]['output']}</prediction>"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                results = evaluate(path, Path(directory) / "scores")
            self.assertEqual(len(results), 9)
            summary = json.loads((Path(directory) / "scores/pilot_scores.summary.json").read_text())
            for mode in ("strict", "relaxed", "normalized"):
                group = summary["summary"][f"cruxeval/{mode}"]
                self.assertEqual((group["total"], group["correct"], group["unscored"]), (3, 1, 2))
                self.assertEqual(group["statuses"], {"scored": 1, "request_error": 1, "request_incomplete": 1})


if __name__ == "__main__":
    unittest.main()
