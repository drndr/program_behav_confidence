"""Confidence metrics: AUROC (also within strata), bootstrap intervals, selective accuracy and calibration.

Standard library only. Every function takes one confidence score per answer and whether
that answer was correct. A higher score must mean more confident that the answer is
correct; negate signals where more means less confident, such as reasoning effort.
Dropping answers without a score is the caller's job.
"""

import math
import random


def _check(scores, correct):
    scores, correct = list(scores), [bool(c) for c in correct]
    if len(scores) != len(correct):
        raise ValueError("scores and correct differ in length")
    if any(not isinstance(s, (int, float)) or math.isnan(s) for s in scores):
        raise ValueError("scores must be numbers, not None or NaN")
    return scores, correct


def auroc(scores, correct):
    """Probability that a correct answer outscores an incorrect one; ties count half.

    Computed from average ranks (Mann-Whitney U). None when there are no correct or no
    incorrect answers.
    """
    scores, correct = _check(scores, correct)
    positives = sum(correct)
    negatives = len(correct) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum, start = 0.0, 0
    while start < len(order):
        end = start
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        rank_sum += (start + end + 1) / 2 * sum(correct[i] for i in order[start:end])  # average of ranks start+1..end
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def stratified_auroc(scores, correct, strata):
    """AUROC over pairs of a correct and an incorrect answer from the same stratum; ties count half.

    The pooled within-stratum AUROC: each stratum's AUROC weighted by its number of such
    pairs, so a score that only tells strata apart scores 0.5. Strata holding only correct
    or only incorrect answers add no pairs. None when no stratum holds both.
    """
    scores, correct = _check(scores, correct)
    if len(strata) != len(scores):
        raise ValueError("strata and scores differ in length")
    groups = {}
    for score, right, stratum in zip(scores, correct, strata):
        groups.setdefault(stratum, []).append((score, right))
    total = pairs = 0
    for members in groups.values():
        value = auroc([s for s, _ in members], [c for _, c in members])
        if value is not None:
            positives = sum(c for _, c in members)
            weight = positives * (len(members) - positives)
            total += value * weight
            pairs += weight
    return total / pairs if pairs else None


def _quantile(ordered, q):
    position = q * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_interval(statistic, *columns, resamples=1000, level=0.95, seed=0):
    """Percentile interval of statistic(*columns) over resamples of the items, with replacement.

    The columns are parallel lists, one entry per item, resampled together. Resamples where
    the statistic is undefined (None) are dropped; the interval is None when fewer than half
    remain. The seed is fixed, so intervals are reproducible.
    """
    n = len(columns[0])
    if not n or any(len(column) != n for column in columns):
        raise ValueError("columns must be non-empty and of equal length")
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        value = statistic(*([column[i] for i in picks] for column in columns))
        if value is not None:
            values.append(value)
    if len(values) < resamples / 2:
        return None
    values.sort()
    tail = (1 - level) / 2
    return _quantile(values, tail), _quantile(values, 1 - tail)


def selective_accuracy(scores, correct, coverage):
    """Accuracy over the most confident `coverage` share of answers (0 < coverage <= 1).

    Answers tied at the cut-off count in proportion: the expected accuracy over random
    tie-breaking, so many equal scores (e.g. a stated confidence of 99) are not ordered
    arbitrarily. None for no answers.
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    scores, correct = _check(scores, correct)
    if not scores:
        return None
    pairs = sorted(zip(scores, correct), key=lambda pair: pair[0], reverse=True)
    keep, taken, right, start = coverage * len(pairs), 0.0, 0.0, 0
    while start < len(pairs) and taken < keep:
        end = start
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        share = min(end - start, keep - taken)
        right += share * sum(c for _, c in pairs[start:end]) / (end - start)
        taken += share
        start = end
    return right / taken


def calibration(probabilities, correct, bins=10):
    """Expected calibration error and its reliability table, over equal-width bins of [0, 1].

    The error is the count-weighted mean of |accuracy - mean probability| across bins. The
    table lists only non-empty bins. ECE is None for no answers.
    """
    probabilities, correct = _check(probabilities, correct)
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("probabilities must lie in [0, 1]")
    groups = {}
    for p, c in zip(probabilities, correct):
        groups.setdefault(min(int(p * bins), bins - 1), []).append((p, c))
    table = [{"low": index / bins, "high": (index + 1) / bins, "count": len(members),
              "mean_probability": sum(p for p, _ in members) / len(members),
              "accuracy": sum(c for _, c in members) / len(members)}
             for index, members in sorted(groups.items())]
    total = sum(row["count"] for row in table)
    ece = sum(row["count"] * abs(row["accuracy"] - row["mean_probability"]) for row in table) / total if total else None
    return {"ece": ece, "bins": table}
