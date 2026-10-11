from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from confidence import summarize


def row(dataset, correct, logprob, code_tokens, source="CRUXEval", reasoning=0, confidence=None):
    return {"dataset": dataset, "source": source, "correct_normalized": correct, "logprob": logprob,
            "reasoning_tokens": reasoning, "completion_tokens": 20, "code_tokens": code_tokens,
            "cyclomatic_complexity": None, "target_cyclomatic_complexity": None,
            "halstead_difficulty": None, "target_halstead_difficulty": None, "information_flow": None,
            "trace_length": None,
            "p_correct": None, "confidence": confidence, "verdict": None}


class SummaryTests(unittest.TestCase):
    def test_groups_signals_and_unscored_answers(self):
        rows = [row("cruxeval", True, -0.01, 10, confidence=99), row("cruxeval", True, -0.02, 20, confidence=99),
                row("cruxeval", False, -0.5, 30, confidence=90), row("cruxeval", None, None, 40),
                row("re2", True, -0.1, 50, source="HumanEval"), row("re2", False, -0.9, 900, source="Real")]
        summary = {entry["group"]: entry for entry in summarize(rows, resamples=50)}
        self.assertEqual(set(summary), {"CRUXEval", "RE2", "RE2 exercises", "RE2 library code"})
        self.assertEqual(summary["RE2 exercises"]["signals"], [])  # one answer: nothing varies
        crux = summary["CRUXEval"]
        self.assertEqual((crux["items"], crux["unscored"], crux["wrong"]), (4, 1, 1))
        self.assertAlmostEqual(crux["accuracy"], 2 / 3)
        signals = {s["signal"]: s for s in crux["signals"]}
        # Zero reasoning tokens and identical generated tokens do not vary, so they are skipped.
        self.assertEqual(set(signals), {"answer logprob", "stated confidence", "code tokens (baseline)"})
        self.assertEqual((signals["answer logprob"]["auroc"], signals["answer logprob"]["n"]), (1.0, 3))
        self.assertEqual(signals["answer logprob"]["selective_accuracy"]["0.5"], 1.0)
        self.assertAlmostEqual(signals["stated confidence"]["ece"], abs(2 / 3 - (0.99 + 0.99 + 0.90) / 3))
        self.assertIsNone(signals["answer logprob"]["ece"])


if __name__ == "__main__":
    unittest.main()
