from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from confidence import stated_correctness, summarize


def row(dataset, correct, logprob, code_tokens, source="CRUXEval", reasoning=0, confidence=None, verdict=None,
        solvable=None):
    return {"dataset": dataset, "source": source, "solvable": solvable, "correct_normalized": correct, "logprob": logprob,
            "reasoning_tokens": reasoning, "completion_tokens": 20, "code_tokens": code_tokens, "input_tokens": 7,
            "cyclomatic_complexity": None, "target_cyclomatic_complexity": None,
            "halstead_difficulty": None, "target_halstead_difficulty": None, "information_flow": None,
            "trace_length": None,
            "p_correct": None, "confidence": confidence, "verdict": verdict}


def signal(entry, name):
    return next(s for s in entry["signals"] if s["signal"] == name)


class StatedCorrectnessTests(unittest.TestCase):
    def test_confidence_in_the_verdict_becomes_probability_the_answer_is_correct(self):
        self.assertEqual(stated_correctness(row("re2", True, None, 1, confidence=80, verdict="correct")), 80)
        self.assertEqual(stated_correctness(row("re2", True, None, 1, confidence=80, verdict="incorrect")), 20)
        self.assertIsNone(stated_correctness(row("re2", True, None, 1, confidence=80)))
        self.assertIsNone(stated_correctness(row("re2", True, None, 1, verdict="correct")))


class SummaryTests(unittest.TestCase):
    def test_groups_signals_and_unscored_answers(self):
        rows = [row("cruxeval", True, -0.01, 10, confidence=99, verdict="correct"),
                row("cruxeval", True, -0.02, 20, confidence=99, verdict="correct"),
                row("cruxeval", False, -0.5, 30, confidence=90, verdict="incorrect"), row("cruxeval", None, None, 40),
                row("re2", True, -0.1, 50, source="HumanEval"), row("re2", False, -0.9, 900, source="Real")]
        summary = {entry["group"]: entry for entry in summarize(rows, resamples=50)}
        self.assertEqual(set(summary), {"CRUXEval", "RE2", "RE2 exercises", "RE2 library code"})
        self.assertEqual(summary["RE2 exercises"]["signals"], [])  # one answer: nothing varies
        crux = summary["CRUXEval"]
        self.assertEqual((crux["items"], crux["unscored"], crux["analysed"], crux["wrong"]), (4, 1, 3, 1))
        self.assertAlmostEqual(crux["accuracy"], 2 / 3)
        signals = {s["signal"]: s for s in crux["signals"]}
        # Zero reasoning tokens, identical generated and input tokens do not vary, so they are skipped.
        self.assertEqual(set(signals), {"answer logprob", "stated confidence", "stated confidence (as written)", "verdict",
                                        "code tokens (baseline)"})
        self.assertEqual((signals["answer logprob"]["auroc"], signals["answer logprob"]["n"]), (1.0, 3))
        self.assertEqual(signals["answer logprob"]["selective_accuracy"]["0.5"], 1.0)
        # "incorrect" at 90 means 10% that the answer is correct.
        self.assertAlmostEqual(signals["stated confidence"]["ece"], (2 * 0.01 + 0.10) / 3)
        self.assertAlmostEqual(signals["stated confidence (as written)"]["ece"], abs(2 / 3 - (0.99 + 0.99 + 0.90) / 3))
        self.assertIsNone(signals["answer logprob"]["ece"])

    def test_solvable_groups_leave_out_items_the_audit_found_not_solvable(self):
        rows = [row("re2", True, -0.1, 10, source="Real", solvable=True),
                row("re2", False, -0.9, 20, source="Real", solvable=True),
                row("re2", False, -0.5, 30, source="Real", solvable=False),
                row("re2", True, -0.2, 40, source="HumanEval", solvable=True)]
        summary = {entry["group"]: entry for entry in summarize(rows, resamples=50)}
        self.assertEqual((summary["RE2 library code"]["items"], summary["RE2 library code solvable"]["items"]), (3, 2))
        self.assertEqual(summary["RE2 solvable"]["items"], 3)
        self.assertAlmostEqual(summary["RE2 library code solvable"]["accuracy"], 1 / 2)
        # Without an audit (solvable None) the solvable groups are empty and left out.
        unaudited = summarize([dict(r, solvable=None) for r in rows], resamples=50)
        self.assertNotIn("RE2 solvable", {entry["group"] for entry in unaudited})

    def test_auroc_within_code_size_tertiles(self):
        # Sizes 1-9 fall into tertiles {1-3}, {4-6}, {7-9}; the log-probability only tells the tertiles apart.
        correct = [True, True, False, True, False, False, True, False, False]
        rows = [row("cruxeval", c, -float(size > 3) - float(size > 6), size) for size, c in zip(range(1, 10), correct)]
        crux = summarize(rows, groups={"CRUXEval": lambda r: True}, resamples=50)[0]
        self.assertGreater(signal(crux, "answer logprob")["auroc"], 0.5)
        self.assertEqual(signal(crux, "answer logprob")["within_size_tertiles"], 0.5)
        plain = summarize(rows, groups={"CRUXEval": lambda r: True}, resamples=50, within_tertiles=False)[0]
        self.assertIsNone(signal(plain, "answer logprob")["within_size_tertiles"])

    def test_auroc_within_source_and_size(self):
        # The log-probability only tells exercises (more often right) from repository code; sizes 1-6 in each.
        exercise = [True, True, True, True, False, True]
        repository = [False, False, True, False, False, False]
        rows = ([row("re2", c, -0.1, size, source="HumanEval") for size, c in zip(range(1, 7), exercise)]
                + [row("re2", c, -0.9, size, source="Real") for size, c in zip(range(1, 7), repository)])
        re2 = summarize(rows, {"RE2": lambda r: True}, resamples=50)[0]
        self.assertGreater(signal(re2, "answer logprob")["within_size_tertiles"], 0.5)
        self.assertEqual(signal(re2, "answer logprob")["within_source_and_size"], 0.5)

    def test_common_answers_and_all_requests(self):
        # Two wrong answers have no verdict (no prediction tag); a truncated request has no answer.
        rows = [row("re2", True, -0.1, 10, confidence=90, verdict="correct"),
                row("re2", True, -0.2, 20, confidence=80, verdict="correct"),
                row("re2", False, -0.3, 30, confidence=70, verdict="correct"),
                row("re2", False, -0.9, 40), row("re2", False, -0.8, 50), row("re2", None, None, 60)]
        group = {"RE2": lambda r: True}
        own = summarize(rows, group, resamples=50)[0]
        self.assertEqual((signal(own, "answer logprob")["n"], signal(own, "stated confidence")["n"]), (5, 3))
        same = summarize(rows, group, resamples=50, signals=["answer logprob", "stated confidence"], common=True)[0]
        self.assertEqual((same["analysed"], same["wrong"]), (3, 1))
        self.assertEqual({s["signal"]: s["n"] for s in same["signals"]}, {"answer logprob": 3, "stated confidence": 3})
        self.assertIsNone(signal(same, "stated confidence")["all_requests"])
        # Over all six requests, the answers without a verdict rank last for stated confidence and are wrong.
        everything = signal(own, "stated confidence")["all_requests"]
        self.assertEqual(everything["n"], 6)
        self.assertEqual(everything["selective_accuracy"]["0.5"], 2 / 3)
        self.assertAlmostEqual(everything["auroc"], (4 + 4) / 8)  # each right answer beats the 4 wrong requests


if __name__ == "__main__":
    unittest.main()
