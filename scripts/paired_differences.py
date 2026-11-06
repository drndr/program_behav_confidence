"""Paired AUROC differences on RE2's solvable items (no model calls): python scripts/paired_differences.py

For every study condition, the difference between two quantities in their AUROC within source
and size, with a 95% percentile bootstrap interval. Both are compared on the same answers, with
the same resamples, so the interval reflects the difference itself. Writes
data/analysis/paired.csv:

  properties  each complexity metric minus code size, on the answers that have every static
              property
  signals     reasoning tokens, stated confidence and answer log-probability minus code size,
              and reasoning tokens minus stated confidence, on the answers that have every
              confidence signal the condition provides

Strata are cut once on the compared answers (confidence.source_size_strata) and not recut in
each resample. Unscored answers are left out, and a pair is skipped where a quantity is missing
or does not vary.
"""

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from confidence import CONFIDENCE_SIGNALS, GROUPS, SIGNALS, STATIC_PROPERTIES, source_size_strata
from metrics import bootstrap_interval, stratified_auroc
from runs import CONDITIONS, condition_rows

SIZE = "code tokens (baseline)"
PROPERTY_PAIRS = [(name, SIZE) for name in ("complexity (baseline)", "Halstead difficulty (baseline)",
                                            "information flow (baseline)")]
SIGNAL_PAIRS = [("reasoning tokens", SIZE), ("stated confidence", SIZE), ("answer logprob", SIZE),
                ("reasoning tokens", "stated confidence")]
COMPARISONS = (("properties", STATIC_PROPERTIES, PROPERTY_PAIRS),
               ("signals", CONFIDENCE_SIGNALS + (SIZE,), SIGNAL_PAIRS))
FIELDS = ["model", "reasoning", "comparison", "signal", "baseline", "answers", "wrong", "difference", "low", "high",
          "excludes_zero"]


def difference(first, second, correct, strata):
    """AUROC of the first scores minus that of the second, within strata; None where either is undefined."""
    a, b = stratified_auroc(first, correct, strata), stratified_auroc(second, correct, strata)
    return None if a is None or b is None else a - b


def paired_differences(rows, shared, pairs, resamples=1000):
    """One record per pair, on the scored rows that have every quantity in `shared` any of them has."""
    scored = [row for row in rows if row["correct_normalized"] is not None]
    present = [name for name in shared if any(SIGNALS[name](row) is not None for row in scored)]
    answers = [row for row in scored if all(SIGNALS[name](row) is not None for name in present)]
    strata = source_size_strata(answers)
    correct = [row["correct_normalized"] for row in answers]
    records = []
    for first, second in pairs:
        if first not in present or second not in present:
            continue
        a, b = [SIGNALS[first](row) for row in answers], [SIGNALS[second](row) for row in answers]
        if len(set(a)) < 2 or len(set(b)) < 2:
            continue
        interval = bootstrap_interval(difference, a, b, correct, strata, resamples=resamples)
        low, high = interval or (None, None)
        records.append({"signal": first, "baseline": second, "answers": len(answers), "wrong": sum(not c for c in correct),
                        "difference": difference(a, b, correct, strata), "low": low, "high": high,
                        "excludes_zero": None if interval is None else low > 0 or high < 0})
    return records


def main(data_dir=ROOT / "data", resamples=1000):
    table = []
    for spec in CONDITIONS:
        rows = [row for row in condition_rows(spec["runs"], data_dir) if GROUPS["RE2 solvable"](row)]
        for comparison, shared, pairs in COMPARISONS:
            table += [{"model": spec["model"], "reasoning": spec["reasoning"], "comparison": comparison, **record}
                      for record in paired_differences(rows, shared, pairs, resamples)]
    with (data_dir / "analysis" / "paired.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        print(f"{row['model']:20} {row['reasoning']:6} {row['signal'][:20]:20} - {row['baseline'][:18]:18} "
              f"{row['difference']:+.2f} [{row['low']:+.2f}, {row['high']:+.2f}]{'  excludes 0' if row['excludes_zero'] else ''}")


if __name__ == "__main__":
    main()
