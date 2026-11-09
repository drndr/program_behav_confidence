"""Fixed-budget error triage on RE2's solvable items (no model calls): python scripts/error_triage.py

For every study condition and signal, the share of errors caught by checking only the 10% and
20% of answers that the signal ranks as most likely to be wrong, rounded down to whole answers.
Signals are compared on the answers that have all of them, and unscored answers are left out.
Tied answers share the budget
evenly, which is the expected result of breaking ties at random, so checking at random catches
the budget itself. The ranking covers all answers, not answers of similar source and size.

Writes data/analysis/triage.csv with, per condition, signal and budget, the errors caught and
the most any ranking could catch at that budget.
"""

import csv
from itertools import groupby
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from confidence import GROUPS, SIGNALS
from runs import CONDITIONS, condition_rows

BUDGETS = (0.1, 0.2)
TRIAGE_SIGNALS = ("reasoning tokens", "generated tokens", "answer logprob", "stated confidence", "verdict",
                  "code tokens (baseline)")
FIELDS = ["model", "reasoning", "answers", "errors", "signal", "budget", "errors_caught", "most_possible"]


def checked(budget, answers):
    """How many answers a budget share allows checking, rounded down to whole answers."""
    return math.floor(budget * answers + 1e-9)


def errors_caught(scores, correct, budget):
    """Share of the errors among the lowest-scoring `budget` share of answers; None without errors."""
    errors = sum(not right for right in correct)
    if not errors:
        return None
    room, found = checked(budget, len(scores)), 0.0
    ranked = sorted(zip(scores, correct), key=lambda pair: pair[0])
    for _, tied in groupby(ranked, key=lambda pair: pair[0]):
        tied = list(tied)
        taken = min(room, len(tied))
        found += sum(not right for _, right in tied) * taken / len(tied)
        room -= taken
        if room <= 0:
            break
    return found / errors


def condition_triage(spec, data_dir):
    """Rows of the triage table for one condition, on the answers that have every signal it provides."""
    rows = [row for row in condition_rows(spec["runs"], data_dir)
            if GROUPS["RE2 solvable"](row) and row["correct_normalized"] is not None]
    present = [name for name in TRIAGE_SIGNALS if len({SIGNALS[name](row) for row in rows} - {None}) > 1]
    answers = [row for row in rows if all(SIGNALS[name](row) is not None for name in present)]
    correct = [row["correct_normalized"] for row in answers]
    errors = sum(not right for right in correct)
    return [{"model": spec["model"], "reasoning": spec["reasoning"], "answers": len(answers), "errors": errors,
             "signal": name, "budget": budget,
             "errors_caught": errors_caught([SIGNALS[name](row) for row in answers], correct, budget),
             "most_possible": min(1.0, checked(budget, len(answers)) / errors)}
            for name in present for budget in BUDGETS]


def main(data_dir=ROOT / "data"):
    table = [row for spec in CONDITIONS for row in condition_triage(spec, data_dir)]
    with (data_dir / "analysis" / "triage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        print(f"{row['model']:20} {row['reasoning']:6} {row['signal']:24} check {row['budget']:.0%}: "
              f"catches {row['errors_caught']:.0%} of {row['errors']} errors (at most {row['most_possible']:.0%})")


if __name__ == "__main__":
    main()
