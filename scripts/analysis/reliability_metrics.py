"""Censoring-aware and repeatability-aware evaluation summaries."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class CensoringSummary:
    total: int
    completed: int
    truncated: int
    correct_total: int
    correct_completed: int
    correct_truncated: int
    unresolved_truncated: int
    completion_rate: float
    system_accuracy: float
    completed_only_accuracy: float | None
    censoring_lower_bound: float
    censoring_upper_bound: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ItemRepeatability:
    item_id: str
    repeats: int
    distinct_answers: int
    modal_answer: str
    modal_count: int
    modal_share: float
    pairwise_disagreement: float
    correct_repeats: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_censoring(
    records: Iterable[dict[str, Any]],
    *,
    correct_field: str = "typed_semantic_correct",
    truncated_field: str = "is_truncated",
) -> CensoringSummary:
    """Summarize observed accuracy and bounds for completion-censored rows.

    The completed-only statistic is conditional on a selected subset and is not
    interpreted as a bound. The worst/best bounds instead assign every
    unresolved truncated response to incorrect/correct respectively, while
    treating extracted answers already scored correct as observed successes.
    """
    rows = list(records)
    total = len(rows)
    truncated_rows = [row for row in rows if bool(row.get(truncated_field))]
    completed_rows = [row for row in rows if not bool(row.get(truncated_field))]
    correct_total = sum(bool(row.get(correct_field)) for row in rows)
    correct_completed = sum(bool(row.get(correct_field)) for row in completed_rows)
    correct_truncated = sum(bool(row.get(correct_field)) for row in truncated_rows)
    unresolved_truncated = len(truncated_rows) - correct_truncated

    if total:
        completion_rate = len(completed_rows) / total
        system_accuracy = correct_total / total
        lower = correct_total / total
        upper = (correct_total + unresolved_truncated) / total
    else:
        completion_rate = 0.0
        system_accuracy = 0.0
        lower = 0.0
        upper = 1.0
    completed_only = correct_completed / len(completed_rows) if completed_rows else None
    return CensoringSummary(
        total=total,
        completed=len(completed_rows),
        truncated=len(truncated_rows),
        correct_total=correct_total,
        correct_completed=correct_completed,
        correct_truncated=correct_truncated,
        unresolved_truncated=unresolved_truncated,
        completion_rate=completion_rate,
        system_accuracy=system_accuracy,
        completed_only_accuracy=completed_only,
        censoring_lower_bound=lower,
        censoring_upper_bound=upper,
    )


def group_censoring_summaries(
    records: Iterable[dict[str, Any]],
    group_fields: Sequence[str],
    *,
    correct_field: str = "typed_semantic_correct",
    truncated_field: str = "is_truncated",
) -> list[dict[str, Any]]:
    """Return deterministic group-level censoring summaries."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(field) for field in group_fields)].append(record)
    output = []
    for key in sorted(groups, key=lambda value: tuple(str(part) for part in value)):
        prefix = dict(zip(group_fields, key))
        prefix.update(
            summarize_censoring(
                groups[key], correct_field=correct_field, truncated_field=truncated_field
            ).to_dict()
        )
        output.append(prefix)
    return output


def _pairwise_disagreement(answers: Sequence[str]) -> float:
    pairs = list(combinations(answers, 2))
    if not pairs:
        return 0.0
    return sum(left != right for left, right in pairs) / len(pairs)


def summarize_item_repeats(
    records: Iterable[dict[str, Any]],
    *,
    id_field: str = "id",
    answer_field: str = "prediction_clean",
    correct_field: str | None = "typed_semantic_correct",
) -> list[ItemRepeatability]:
    """Compute item-level modal agreement and pairwise disagreement."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(id_field, ""))].append(record)

    summaries = []
    for item_id in sorted(groups):
        rows = groups[item_id]
        answers = [str(row.get(answer_field, "")) for row in rows]
        counts = Counter(answers)
        modal_answer, modal_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        correct_repeats = None
        if correct_field is not None:
            correct_repeats = sum(bool(row.get(correct_field)) for row in rows)
        summaries.append(
            ItemRepeatability(
                item_id=item_id,
                repeats=len(rows),
                distinct_answers=len(counts),
                modal_answer=modal_answer,
                modal_count=modal_count,
                modal_share=modal_count / len(rows),
                pairwise_disagreement=_pairwise_disagreement(answers),
                correct_repeats=correct_repeats,
            )
        )
    return summaries


def answer_entropy(answers: Sequence[str]) -> float:
    """Shannon entropy in bits for a set of repeated answers."""
    if not answers:
        return 0.0
    counts = Counter(answers)
    total = len(answers)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())
