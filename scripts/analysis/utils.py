"""Dataset-agnostic helpers shared by the analysis scripts.

Dataset-specific rules stay in the per-dataset modules; this module holds only
what every track needs: prediction-text normalization, JSONL reading and
writing, and the Wilson interval used to report accuracy.
"""

from __future__ import annotations

import ast
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any


EVAL_FAIL = object()
DEFAULT_Z = 1.959963984540054


def literal_eval_safe(value: Any) -> Any:
    """Parse a Python literal, returning EVAL_FAIL when the text does not parse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            return ast.literal_eval(str(value).strip())
        except Exception:
            return EVAL_FAIL


def strip_fence_lines(text: str) -> str:
    """Drop a leading fence line, with any language tag, and a trailing fence line."""
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def strip_wrapping_fence(value: str) -> str:
    """Strip one wrapping Markdown fence from raw predicted text.

    A trailing fence is removed only when the text opens with a fence, which
    keeps unfenced console output, where a line of backticks could be real
    content, untouched.
    """
    text = str(value or "").strip()
    if not text.startswith("```"):
        return text
    return strip_fence_lines(text)


def clean_prediction(value: str) -> str:
    """Normalize a predicted Python literal for comparison against a target.

    Removes a chain-of-thought marker, one wrapping Markdown fence including a
    dangling closing fence, and inline backticks. Language tags are matched by
    position rather than by an allowlist, so `r` and `output` behave like
    `python`.
    """
    text = str(value or "").strip()
    if text.lower().startswith("final_output:"):
        text = text.split(":", 1)[1].strip()
    if len(text.splitlines()) > 1:
        text = strip_fence_lines(text)
    if len(text.splitlines()) <= 1 and len(text) > 1:
        if text.startswith("`") or text.endswith("`"):
            text = text.strip("`").strip()
    return text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, raising on the first malformed line."""
    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
    return records


def read_jsonl_with_errors(path: Path) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    """Read JSONL, collecting malformed lines instead of raising.

    Used by the scripts that report a malformed-line count rather than failing,
    so a partially written run can still be inspected.
    """
    records = []
    errors = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append((line_number, str(exc)))
    return records, errors


def write_jsonl(path: Path, records: list[dict[str, Any]], sort_keys: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False, sort_keys=sort_keys) + "\n")


def duplicate_ids(records: list[dict[str, Any]]) -> list[str]:
    """Return the IDs appearing more than once, so callers can word their own error."""
    ids = [str(record.get("id", "")) for record in records]
    return [sample_id for sample_id, count in Counter(ids).items() if count > 1]


def wilson_interval(correct: int, total: int, z: float = DEFAULT_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 1.0
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(
        (proportion * (1 - proportion) + z * z / (4 * total)) / total
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)
