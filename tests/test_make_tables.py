from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_tables import accuracy_table, audit_table, auroc_cell, triage_table, within_table

SWE, PROJECT, EXERCISE = "django__django-1@@m.py@@f", "lib@@lib_m.py@@f_L1", "HumanEval_1"


def accuracy_rows(model, reasoning, counts):
    """counts: group -> (correct, items)."""
    return [{"model": model, "reasoning": reasoning, "group": group, "mode": "normalized", "correct": str(right),
             "items": str(items)} for group, (right, items) in counts.items()]


COUNTS = {"CRUXEval": (8, 10), "RE2 solvable": (6, 8), "RE2 library code solvable": (2, 4)}


class AccuracyTableTests(unittest.TestCase):
    def test_solvable_exercises_are_the_solvable_examples_outside_repository_code(self):
        table = accuracy_table(accuracy_rows("GPT-6 Luna", "none", COUNTS))
        self.assertIn(r" & & (10) & (4) & (4) & (8) \\", table)
        self.assertIn(r"Luna & none & \textbf{80.0} & \textbf{100.0} & \textbf{50.0} & \textbf{75.0} \\", table)

    def test_best_per_column_in_bold_and_a_rule_after_a_model_with_several_efforts(self):
        rows = (accuracy_rows("GPT-6 Luna", "none", COUNTS)
                + accuracy_rows("GPT-6 Luna", "low", {"CRUXEval": (9, 10), "RE2 solvable": (5, 8),
                                                      "RE2 library code solvable": (1, 4)})
                + accuracy_rows("Claude Haiku 5.5", "low", COUNTS))
        table = accuracy_table(rows)
        self.assertIn(r"Luna  & none & 80.0 & \textbf{100.0} & \textbf{50.0} & \textbf{75.0} \\", table)
        self.assertIn(r"      & low  & \textbf{90.0} & \textbf{100.0} & 25.0 & 62.5 \\", table)  # model named once
        self.assertEqual(table.count(r"\midrule"), 2)  # after the header and after Luna's block


class AuditTableTests(unittest.TestCase):
    def test_labels_per_source_with_totals(self):
        audit = [{"id": SWE, "solvable": "False", "category": "input_lost"},
                 {"id": PROJECT, "solvable": "False", "category": "environment"},
                 {"id": EXERCISE, "solvable": "True", "category": "missed_computation"}]
        table = audit_table(audit, [SWE, PROJECT, EXERCISE, "HumanEval_2"])
        self.assertIn("the 3 RE2 examples", table)
        self.assertIn(r"Input lost & 0 & 1 & 0 & 1 \\", table)
        self.assertIn(r"Environment-dependent & 1 & 0 & 0 & 1 \\", table)
        self.assertIn(r"Not solvable & 1 & 1 & 0 & 2 \\", table)
        self.assertIn(r"Solvable & 0 & 0 & 1 & 1 \\", table)
        self.assertIn(r"All examples & 1 & 1 & 2 & 4 \\", table)


def signal_row(model, signal, auroc, low, high, group="RE2 solvable"):
    return {"model": model, "reasoning": "low", "group": group, "signal": signal, "comparison": "static",
            "within_source_and_size_auroc": str(auroc), "within_source_and_size_low": str(low),
            "within_source_and_size_high": str(high)}


class WithinTableTests(unittest.TestCase):
    def test_cell_drops_the_leading_zero_of_the_interval(self):
        _, value, interval = auroc_cell(signal_row("M", "x", 0.564, 0.481, 0.632), dagger=True)
        self.assertEqual((value, interval), (r"0.56$^\dagger$", "[.48, .63]"))

    def test_bold_best_dagger_above_size_and_missing_signals(self):
        model = "Claude Haiku 5.5"
        properties = [signal_row(model, name, 0.6, 0.5, 0.7) for name in
                      ("code tokens (baseline)", "complexity (baseline)", "Halstead difficulty (baseline)",
                       "information flow (baseline)")]
        signals = [signal_row(model, "reasoning tokens", 0.67, 0.59, 0.74),
                   signal_row(model, "stated confidence", 0.79, 0.71, 0.85)]
        paired = [{"model": model, "reasoning": "low", "signal": "stated confidence", "baseline": "code tokens (baseline)",
                   "excludes_zero": "True", "difference": "0.2"},
                  {"model": model, "reasoning": "low", "signal": "reasoning tokens", "baseline": "code tokens (baseline)",
                   "excludes_zero": "False", "difference": "0.1"}]
        verification = [{"model": model, "reasoning": "low", "dataset": "re2", "subset": "solvable",
                         "approved_wrong": "59", "rejected_wrong": "41"}]
        table = within_table(properties, signals, paired, verification)
        self.assertIn(r"Haiku, low & 0.60 [.50, .70] & 0.60 [.50, .70] & 0.60 [.50, .70] & 0.60 [.50, .70] & -- & "
                      r"0.67 [.59, .74] & \textbf{0.79$^\dagger$} [.71, .85] & \qty{59}{\percent} \\", table)


class TriageTableTests(unittest.TestCase):
    def test_percent_caught_at_the_budget_with_dashes_for_missing_signals(self):
        triage = [{"model": "GPT-6 Luna", "reasoning": "none", "signal": signal, "budget": budget, "errors_caught": value}
                  for signal, value in (("code tokens (baseline)", "0.342"), ("stated confidence", "0.414"))
                  for budget in ("0.1", "0.2")]
        table = triage_table(triage)
        self.assertIn(r"checking the 20\% of answers", table)
        self.assertIn(r"Luna, none & 34 & -- & 41 & -- \\", table)


if __name__ == "__main__":
    unittest.main()
