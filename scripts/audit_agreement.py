"""Compare reviews of the RE2 audit with the audit and with each other: python scripts/audit_agreement.py REVIEW.csv ...

Each review is a CSV exported from data/audit/review/review.html (scripts/make_audit_review.py).
For every pair of the audit and the reviews, prints on the items both labelled the share they
agree on and Cohen's kappa, for whether an item is solvable as given and for the verdict
(input lost, reference misrecorded, environment-dependent, or solvable); then lists every item
on which they do not all agree, for adjudication.
"""

import argparse
import csv
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from make_audit_review import verdict
from runs import AUDIT


def kappa(first, second):
    """Cohen's kappa of two raters' labels for the same items; None when chance agreement is certain."""
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    expected = sum(first.count(label) / n * second.count(label) / n for label in set(first) | set(second))
    return None if expected == 1 else (observed - expected) / (1 - expected)


def review_verdicts(path):
    """Rater name and verdict per labelled item of an exported review."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["verdict"]]
    rater = rows[0]["rater"] if rows else Path(path).stem
    return rater, {row["id"]: (row["verdict"], row["reason"]) for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reviews", nargs="+", help="CSV files exported from the review page")
    args = parser.parse_args()
    with (ROOT / "data" / AUDIT).open(encoding="utf-8", newline="") as stream:
        sources = {"audit": {row["id"]: (verdict(row), row["note"]) for row in csv.DictReader(stream)}}
    for path in args.reviews:
        rater, labels = review_verdicts(path)
        sources[rater] = labels
    for (name_a, a), (name_b, b) in combinations(sources.items(), 2):
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        for what, label in (("solvable", lambda v: v == "solvable"), ("verdict", lambda v: v)):
            first, second = [label(a[i][0]) for i in shared], [label(b[i][0]) for i in shared]
            agreement = sum(x == y for x, y in zip(first, second)) / len(shared)
            value = kappa(first, second)
            print(f"{name_a} vs {name_b}, {what}: {len(shared)} items, agreement {agreement:.0%}, "
                  f"kappa {'n/a' if value is None else f'{value:.2f}'}")
    reviewed = sorted(set().union(*(set(labels) for name, labels in sources.items() if name != "audit")))
    disputed = [i for i in reviewed if len({labels[i][0] for labels in sources.values() if i in labels}) > 1]
    print(f"\n{len(disputed)} of {len(reviewed)} reviewed items without full agreement:")
    for i in disputed:
        print(f"\n{i}")
        for name, labels in sources.items():
            if i in labels:
                print(f"  {name}: {labels[i][0]}: {labels[i][1]}")


if __name__ == "__main__":
    main()
