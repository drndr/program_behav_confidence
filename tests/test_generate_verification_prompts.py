import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_verification_prompts import generate
from datasets import DATASET_SOURCES, load_examples


def write_run(directory, rows, repetitions=1):
    path = Path(directory) / "pilot.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest = {"configuration": {"model": "org/model", "parameters": {"max_tokens": 1}, "repetitions": repetitions}}
    path.with_name("pilot.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


class VerificationPromptTests(unittest.TestCase):
    def setUp(self):
        self.examples = load_examples(ROOT / "data" / DATASET_SOURCES["cruxeval"], "cruxeval")[:4]
        ids = [e["id"] for e in self.examples]
        self.rows = [
            {"dataset": "cruxeval", "id": ids[0], "repetition": 0, "attempt": 1, "status": "completed",
             "text": "<prediction>[1, 2]</prediction>"},
            {"dataset": "cruxeval", "id": ids[1], "repetition": 0, "attempt": 1, "status": "completed",
             "text": "the answer is 4"},
            {"dataset": "cruxeval", "id": ids[2], "repetition": 0, "attempt": 1, "status": "incomplete", "text": None},
            {"dataset": "cruxeval", "id": ids[3], "repetition": 0, "attempt": 1, "status": "error"},
            # A retry of the first item is the latest attempt and must be the one judged.
            {"dataset": "cruxeval", "id": ids[0], "repetition": 0, "attempt": 2, "status": "completed",
             "text": "<prediction>'retry'</prediction>"},
        ]

    def test_one_prompt_per_judgeable_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_run(directory, self.rows)
            manifests = generate(path, Path(directory) / "verification")
            record = manifests["cruxeval"]
            self.assertEqual((record["count"], record["prompt_kind"], record["wording_final"]), (1, "verification", True))
            self.assertEqual(record["skipped"], {"no_prediction_tag": 1, "request_incomplete": 1, "request_error": 1})
            self.assertEqual(record["prediction_run"]["model"], "org/model")
            rows = [json.loads(l) for l in (Path(directory) / "verification/cruxeval_prompts.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual((row["id"], row["candidate"], row["prediction_attempt"], row["prediction_run"]),
                             (self.examples[0]["id"], "'retry'", 2, "pilot"))
            prompt = row["messages"][0]["content"]
            self.assertIn("[CANDIDATE]\n'retry'\n[/CANDIDATE]", prompt)
            self.assertIn(self.examples[0]["code"], prompt)
            self.assertNotIn(self.examples[0]["output"], prompt.split("[CANDIDATE]")[0])
            self.assertTrue((Path(directory) / "verification/cruxeval_prompts.manifest.json").exists())

    def test_rejects_multi_repetition_runs_for_now(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_run(directory, self.rows, repetitions=2)
            with self.assertRaises(NotImplementedError):
                generate(path, Path(directory) / "verification")


if __name__ == "__main__":
    unittest.main()
