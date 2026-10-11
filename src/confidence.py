"""Confidence signals per answer and their summary, shared by the analysis and the explorer.

Rows come from runs.item_table. Correctness is normalized scoring; answers without a score
(truncated or failed requests) are left out of every metric and counted separately. Each
signal is oriented so that a higher value means more confident that the answer is correct.
A signal is summarised only where it varies, so e.g. zero reasoning tokens are skipped.
"""

from datasets import RE2_LIBRARY_SOURCES
from metrics import auroc, bootstrap_interval, calibration, selective_accuracy


def _negated(field):
    return lambda row: -row[field] if row[field] is not None else None


SIGNALS = {
    "answer logprob": lambda row: row["logprob"],
    "reasoning tokens": _negated("reasoning_tokens"),
    "generated tokens": _negated("completion_tokens"),
    "verdict P(correct)": lambda row: row["p_correct"],
    "stated confidence": lambda row: row["confidence"],
    "verdict": lambda row: None if row["verdict"] is None else int(row["verdict"] == "correct"),
    "code tokens (baseline)": _negated("code_tokens"),
    "complexity (baseline)": _negated("cyclomatic_complexity"),
    "target complexity (baseline)": _negated("target_cyclomatic_complexity"),
    "Halstead difficulty (baseline)": _negated("halstead_difficulty"),
    "target Halstead difficulty (baseline)": _negated("target_halstead_difficulty"),
    "information flow (baseline)": _negated("information_flow"),
    "trace length (baseline)": _negated("trace_length"),
}
# Signals that are probabilities of correctness, for calibration.
PROBABILITIES = {"verdict P(correct)": lambda row: row["p_correct"],
                 "stated confidence": lambda row: None if row["confidence"] is None else row["confidence"] / 100}
GROUPS = {
    "CRUXEval": lambda row: row["dataset"] == "cruxeval",
    "RE2": lambda row: row["dataset"] == "re2",
    "RE2 exercises": lambda row: row["dataset"] == "re2" and row["source"] not in RE2_LIBRARY_SOURCES,
    "RE2 library code": lambda row: row["dataset"] == "re2" and row["source"] in RE2_LIBRARY_SOURCES,
}
COVERAGES = (0.8, 0.5)


def summarize(rows, groups=GROUPS, resamples=1000):
    """Per group: answers, accuracy, and per signal AUROC with a 95% interval, selective
    accuracy at each coverage, and calibration error for probability signals."""
    summary = []
    for group, member in groups.items():
        members = [row for row in rows if member(row)]
        scored = [row for row in members if row["correct_normalized"] is not None]
        if not scored:
            continue
        entry = {"group": group, "items": len(members), "unscored": len(members) - len(scored),
                 "accuracy": sum(row["correct_normalized"] for row in scored) / len(scored),
                 "wrong": sum(not row["correct_normalized"] for row in scored), "signals": []}
        for name, signal in SIGNALS.items():
            pairs = [(signal(row), row["correct_normalized"]) for row in scored if signal(row) is not None]
            if len({score for score, _ in pairs}) < 2:
                continue
            scores, correct = [s for s, _ in pairs], [c for _, c in pairs]
            result = {"signal": name, "n": len(pairs), "auroc": auroc(scores, correct),
                      "interval": bootstrap_interval(auroc, scores, correct, resamples=resamples),
                      "selective_accuracy": {str(c): selective_accuracy(scores, correct, c) for c in COVERAGES},
                      "ece": None}
            if name in PROBABILITIES:
                probabilities = [PROBABILITIES[name](row) for row in scored if signal(row) is not None]
                result["ece"] = calibration(probabilities, correct)["ece"]
            entry["signals"].append(result)
        summary.append(entry)
    return summary
