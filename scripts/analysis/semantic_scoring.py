"""Auditable scoring primitives for program-behavior predictions.

The module separates four questions that older analysis code combined:

1. Does the extracted text exactly match the reference serialization?
2. Do both strings denote recursively type-identical Python values?
3. Is a bare, unquoted string equal to a string-valued reference?
4. Was the response truncated before or after any extracted answer?

The typed scorer is deliberately conservative. Dataset-specific equivalences,
such as floating-point tolerances or ecosystem-specific printing rules, belong
in separately validated adapters rather than in this shared core.
"""

from __future__ import annotations

import ast
import warnings
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable

from utils import clean_prediction


class SemanticStatus(str, Enum):
    STRICT_MATCH = "strict_match"
    TYPED_MATCH = "typed_match"
    BARE_STRING_MATCH = "bare_string_match"
    EMPTY_PREDICTION = "empty_prediction"
    TARGET_PARSE_ERROR = "target_parse_error"
    PREDICTION_PARSE_ERROR = "prediction_parse_error"
    TYPE_MISMATCH = "type_mismatch"
    WRONG_VALUE = "wrong_value"


TRUNCATION_REASONS = {
    "length",
    "max_length",
    "max_tokens",
    "max_output_tokens",
}


@dataclass(frozen=True)
class ParseResult:
    parsed: bool
    value: Any = None
    error_type: str | None = None


@dataclass(frozen=True)
class SemanticScore:
    strict_correct: bool
    typed_semantic_correct: bool
    bare_string_content_correct: bool
    semantic_status: str
    prediction_clean: str
    target_clean: str
    prediction_parseable: bool
    target_parseable: bool
    prediction_type: str | None
    target_type: str | None
    finish_reason: str
    native_finish_reason: str
    is_truncated: bool
    has_extracted_answer: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_python_literal(value: Any) -> ParseResult:
    """Parse one Python literal without executing arbitrary code."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            return ParseResult(parsed=True, value=ast.literal_eval(str(value).strip()))
        except Exception as exc:
            return ParseResult(parsed=False, error_type=type(exc).__name__)


def _unordered_typed_equal(left: Iterable[Any], right: Iterable[Any]) -> bool:
    """Compare unordered containers without Python's bool/int key coercion."""
    unmatched = list(right)
    for left_value in left:
        for index, right_value in enumerate(unmatched):
            if recursively_typed_equal(left_value, right_value):
                unmatched.pop(index)
                break
        else:
            return False
    return not unmatched


def recursively_typed_equal(left: Any, right: Any) -> bool:
    """Return equality only when recursive Python types and values agree."""
    if type(left) is not type(right):
        return False
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            recursively_typed_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, dict):
        if len(left) != len(right):
            return False
        return _unordered_typed_equal(left.items(), right.items())
    if isinstance(left, (set, frozenset)):
        return len(left) == len(right) and _unordered_typed_equal(left, right)
    return bool(left == right)


def is_truncated_response(finish_reason: Any = "", native_finish_reason: Any = "") -> bool:
    """Recognize the completion-limit labels used by local and API runners."""
    reasons = {str(finish_reason or "").strip().lower(), str(native_finish_reason or "").strip().lower()}
    return bool(reasons & TRUNCATION_REASONS)


def score_python_literal(
    prediction: Any,
    target: Any,
    *,
    finish_reason: Any = "",
    native_finish_reason: Any = "",
) -> SemanticScore:
    """Score an extracted prediction against a Python-literal reference."""
    prediction_clean = clean_prediction(str(prediction or ""))
    target_clean = str(target).strip()
    strict_correct = prediction_clean == target_clean
    prediction_parse = parse_python_literal(prediction_clean) if prediction_clean else ParseResult(False)
    target_parse = parse_python_literal(target_clean)

    typed_correct = bool(
        prediction_parse.parsed
        and target_parse.parsed
        and recursively_typed_equal(prediction_parse.value, target_parse.value)
    )
    bare_string_correct = bool(
        typed_correct
        or (
            target_parse.parsed
            and isinstance(target_parse.value, str)
            and not prediction_parse.parsed
            and prediction_clean == target_parse.value
        )
    )

    if strict_correct:
        status = SemanticStatus.STRICT_MATCH
    elif typed_correct:
        status = SemanticStatus.TYPED_MATCH
    elif bare_string_correct:
        status = SemanticStatus.BARE_STRING_MATCH
    elif not prediction_clean:
        status = SemanticStatus.EMPTY_PREDICTION
    elif not target_parse.parsed:
        status = SemanticStatus.TARGET_PARSE_ERROR
    elif not prediction_parse.parsed:
        status = SemanticStatus.PREDICTION_PARSE_ERROR
    elif type(prediction_parse.value) is not type(target_parse.value):
        status = SemanticStatus.TYPE_MISMATCH
    else:
        status = SemanticStatus.WRONG_VALUE

    return SemanticScore(
        strict_correct=strict_correct,
        typed_semantic_correct=typed_correct,
        bare_string_content_correct=bare_string_correct,
        semantic_status=status.value,
        prediction_clean=prediction_clean,
        target_clean=target_clean,
        prediction_parseable=prediction_parse.parsed,
        target_parseable=target_parse.parsed,
        prediction_type=type(prediction_parse.value).__name__ if prediction_parse.parsed else None,
        target_type=type(target_parse.value).__name__ if target_parse.parsed else None,
        finish_reason=str(finish_reason or ""),
        native_finish_reason=str(native_finish_reason or ""),
        is_truncated=is_truncated_response(finish_reason, native_finish_reason),
        has_extracted_answer=bool(prediction_clean),
    )

