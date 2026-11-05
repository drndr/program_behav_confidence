"""Confidence signals per answer and their summary, shared by the analysis and the explorer.

Rows come from runs.item_table. Correctness is normalized scoring; answers without a score
(truncated or failed requests) are left out of every metric and counted separately. Groups
cover each dataset, RE2's exercises and repository code, and RE2 again without the items an
audit found not solvable as given (lost input state, random or environment-dependent
output, a wrong or differently recorded reference; see runs.AUDIT). Each
signal is oriented so that a higher value means more confident that the answer is correct.
A signal is summarised only where it varies, so e.g. zero reasoning tokens are skipped.

Each signal is first measured on every answer that has it. Signals can then miss different
answers (no verdict where the response had no prediction tag, traces only where running the
program reproduced the reference), so a comparison of signals restricts them to the same
answers (common=True). The all-requests view instead keeps every request of the group:
requests without an answer count as wrong and a missing signal as the least confidence.

The AUROC within code-size tertiles compares answers only with answers of similar size; size
still varies within a tertile, so it narrows rather than removes what size explains. The AUROC
within source and size also keeps the source apart (CRUXEval, RE2 exercises, RE2 repository
code), with tertiles taken within each source, so a signal that only tells repository code from
exercises gains nothing.

Stated confidence is the judge's confidence in its own verdict, not in the answer: it never
falls below 50 after an "incorrect" verdict, even with wording that says it describes the
candidate whatever the verdict (pilot, 15 Oct 2026). It is turned into the probability that the
answer is correct (stated_correctness); the number as written stays as a sensitivity signal.
"""

import math
import statistics

from datasets import RE2_LIBRARY_SOURCES
from metrics import auroc, bootstrap_interval, calibration, selective_accuracy, stratified_auroc


def _negated(field):
    return lambda row: -row[field] if row[field] is not None else None


def stated_correctness(row):
    """Stated confidence as the probability (0-100) that the answer is correct: as given after a
    "correct" verdict, 100 minus it after "incorrect". None without both."""
    if row["confidence"] is None or row["verdict"] is None:
        return None
    return row["confidence"] if row["verdict"] == "correct" else 100 - row["confidence"]


def _percent(value):
    return None if value is None else value / 100


SIGNALS = {
    "answer logprob": lambda row: row["logprob"],
    "reasoning tokens": _negated("reasoning_tokens"),
    "generated tokens": _negated("completion_tokens"),
    "verdict P(correct)": lambda row: row["p_correct"],
    "stated confidence": stated_correctness,
    "stated confidence (as written)": lambda row: row["confidence"],
    "verdict": lambda row: None if row["verdict"] is None else int(row["verdict"] == "correct"),
    "code tokens (baseline)": _negated("code_tokens"),
    "input tokens (baseline)": _negated("input_tokens"),
    "complexity (baseline)": _negated("cyclomatic_complexity"),
    "target complexity (baseline)": _negated("target_cyclomatic_complexity"),
    "Halstead difficulty (baseline)": _negated("halstead_difficulty"),
    "target Halstead difficulty (baseline)": _negated("target_halstead_difficulty"),
    "information flow (baseline)": _negated("information_flow"),
    "trace length (baseline)": _negated("trace_length"),
}
CONFIDENCE_SIGNALS = ("answer logprob", "reasoning tokens", "generated tokens", "verdict P(correct)",
                      "stated confidence", "stated confidence (as written)", "verdict")
STATIC_PROPERTIES = ("code tokens (baseline)", "input tokens (baseline)", "complexity (baseline)",
                     "target complexity (baseline)", "Halstead difficulty (baseline)",
                     "target Halstead difficulty (baseline)", "information flow (baseline)")
# Signals that are probabilities of correctness, for calibration.
PROBABILITIES = {"verdict P(correct)": lambda row: row["p_correct"],
                 "stated confidence": lambda row: _percent(stated_correctness(row)),
                 "stated confidence (as written)": lambda row: _percent(row["confidence"])}
GROUPS = {
    "CRUXEval": lambda row: row["dataset"] == "cruxeval",
    "RE2": lambda row: row["dataset"] == "re2",
    "RE2 exercises": lambda row: row["dataset"] == "re2" and row["source"] not in RE2_LIBRARY_SOURCES,
    "RE2 library code": lambda row: row["dataset"] == "re2" and row["source"] in RE2_LIBRARY_SOURCES,
    # The same without the items the audit found not solvable as given (runs.AUDIT).
    "RE2 solvable": lambda row: row["dataset"] == "re2" and row["solvable"] is True,
    "RE2 library code solvable": lambda row: (row["dataset"] == "re2" and row["source"] in RE2_LIBRARY_SOURCES
                                              and row["solvable"] is True),
}
COVERAGES = (0.8, 0.5)


def code_size_tertiles(rows):
    """Code-size tertile (0, 1, 2) per row, with cut points at the 1/3 and 2/3 quantiles of the
    rows' code tokens, so tied sizes share a tertile; None for fewer than three distinct sizes."""
    sizes = [row["code_tokens"] for row in rows]
    if len(set(sizes)) < 3:
        return None
    first, second = statistics.quantiles(sizes, n=3, method="inclusive")
    return [0 if size <= first else 1 if size <= second else 2 for size in sizes]


def source_kind(row):
    """CRUXEval, RE2 exercise or RE2 repository code."""
    if row["dataset"] != "re2":
        return row["dataset"]
    return "re2 repository" if row["source"] in RE2_LIBRARY_SOURCES else "re2 exercise"


def source_size_strata(rows):
    """(source kind, code-size tertile within that source) per row; a source with fewer than
    three distinct sizes is one stratum."""
    strata = [None] * len(rows)
    for kind in {source_kind(row) for row in rows}:
        indices = [i for i, row in enumerate(rows) if source_kind(row) == kind]
        tertiles = code_size_tertiles([rows[i] for i in indices]) or [0] * len(indices)
        for index, tertile in zip(indices, tertiles):
            strata[index] = (kind, tertile)
    return strata


def _all_requests(signal, members):
    """AUROC and selective accuracy over every request: no answer counts as wrong, no signal as least confident."""
    scores = [-math.inf if signal(row) is None else signal(row) for row in members]
    correct = [row["correct_normalized"] is True for row in members]
    return {"n": len(members), "auroc": auroc(scores, correct),
            "selective_accuracy": {str(c): selective_accuracy(scores, correct, c) for c in COVERAGES}}


def summarize(rows, groups=GROUPS, resamples=1000, within_tertiles=True, signals=None, common=False):
    """Per group: answers, accuracy, and per signal AUROC with a 95% interval, the same within
    code-size tertiles and within source and size (both skipped with within_tertiles=False),
    selective accuracy at each coverage, calibration error for probability signals, and the
    all-requests view.

    signals names the signals to summarise (all by default). With common=True, each group keeps
    only the answers that have every one of them that any of its answers has, and the
    all-requests view is left out; accuracy is then that of the answers kept.
    """
    chosen = {name: SIGNALS[name] for name in (signals or SIGNALS)}
    summary = []
    for group, member in groups.items():
        members = [row for row in rows if member(row)]
        scored = [row for row in members if row["correct_normalized"] is not None]
        if common:
            present = [signal for signal in chosen.values() if any(signal(row) is not None for row in scored)]
            scored = [row for row in scored if all(signal(row) is not None for signal in present)]
        if not scored:
            continue
        entry = {"group": group, "items": len(members), "unscored": sum(row["correct_normalized"] is None for row in members),
                 "analysed": len(scored), "accuracy": sum(row["correct_normalized"] for row in scored) / len(scored),
                 "wrong": sum(not row["correct_normalized"] for row in scored), "signals": []}
        tertiles = code_size_tertiles(scored) if within_tertiles else None
        source_strata = source_size_strata(scored) if within_tertiles else None
        for name, signal in chosen.items():
            kept = [index for index, row in enumerate(scored) if signal(row) is not None]
            pairs = [(signal(scored[i]), scored[i]["correct_normalized"]) for i in kept]
            if len({score for score, _ in pairs}) < 2:
                continue
            scores, correct = [s for s, _ in pairs], [c for _, c in pairs]
            result = {"signal": name, "n": len(pairs), "auroc": auroc(scores, correct),
                      "interval": bootstrap_interval(auroc, scores, correct, resamples=resamples),
                      "within_size_tertiles": None, "within_size_tertiles_interval": None,
                      "within_source_and_size": None, "within_source_and_size_interval": None,
                      "selective_accuracy": {str(c): selective_accuracy(scores, correct, c) for c in COVERAGES},
                      "ece": None, "all_requests": None if common else _all_requests(signal, members)}
            if tertiles:
                strata = [tertiles[i] for i in kept]
                result["within_size_tertiles"] = stratified_auroc(scores, correct, strata)
                result["within_size_tertiles_interval"] = bootstrap_interval(stratified_auroc, scores, correct, strata,
                                                                             resamples=resamples)
            if source_strata:
                strata = [source_strata[i] for i in kept]
                result["within_source_and_size"] = stratified_auroc(scores, correct, strata)
                result["within_source_and_size_interval"] = bootstrap_interval(stratified_auroc, scores, correct,
                                                                               strata, resamples=resamples)
            if name in PROBABILITIES:
                probabilities = [PROBABILITIES[name](row) for row in scored if signal(row) is not None]
                result["ece"] = calibration(probabilities, correct)["ece"]
            entry["signals"].append(result)
        summary.append(entry)
    return summary
