"""Safe parsing and scoring primitives for RE2-Bench output prediction."""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from semantic_scoring import is_truncated_response, recursively_typed_equal


OUTPUT_TAG = re.compile(r"\[OUTPUT\]", re.IGNORECASE)
OUTPUT_CLOSE_TAG = re.compile(r"\[/OUTPUT\]", re.IGNORECASE)
FENCE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Re2ParseResult:
    parsed: bool
    status: str
    mode: str | None
    value: Any
    extracted_text: str
    repair_count: int
    error: str | None
    has_open_tag: bool
    has_close_tag: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Re2Score:
    strict_json_typed_correct: bool
    repaired_json_typed_correct: bool
    safe_compatibility_correct: bool
    parse_status: str
    parse_mode: str | None
    parse_repair_count: int
    prediction_type: str | None
    target_type: str
    extracted_text: str
    has_output_open_tag: bool
    has_output_close_tag: bool
    finish_reason: str
    native_finish_reason: str
    is_truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _remove_hash_comments(text: str) -> str:
    output = []
    for line in text.splitlines():
        characters = []
        in_string = False
        escaped = False
        for character in line:
            if escaped:
                characters.append(character)
                escaped = False
                continue
            if character == "\\" and in_string:
                characters.append(character)
                escaped = True
                continue
            if character == '"':
                in_string = not in_string
                characters.append(character)
                continue
            if character == "#" and not in_string:
                break
            characters.append(character)
        output.append("".join(characters))
    return "\n".join(output)


def _outside_double_string_mask(text: str) -> list[bool]:
    mask = [True] * len(text)
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if escaped:
            mask[index] = not in_string
            escaped = False
            continue
        if character == "\\" and in_string:
            mask[index] = False
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            mask[index] = False
            continue
        mask[index] = not in_string
    return mask


def _value_span(text: str, error_position: int) -> tuple[int, int] | None:
    """Find the object value containing a JSON parse error without evaluating it."""
    mask = _outside_double_string_mask(text)
    colon = min(max(error_position, 0), len(text) - 1) if text else -1
    while colon >= 0 and not (mask[colon] and text[colon] == ":"):
        colon -= 1
    if colon < 0:
        return None
    start = colon + 1
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        return None

    braces = brackets = parentheses = 0
    in_string = escaped = False
    end = start
    while end < len(text):
        character = text[end]
        if escaped:
            escaped = False
            end += 1
            continue
        if character == "\\" and in_string:
            escaped = True
            end += 1
            continue
        if character == '"':
            in_string = not in_string
            end += 1
            continue
        if not in_string:
            if character == "{":
                braces += 1
            elif character == "}":
                if braces == brackets == parentheses == 0:
                    break
                braces = max(0, braces - 1)
            elif character == "[":
                brackets += 1
            elif character == "]":
                if braces == brackets == parentheses == 0:
                    break
                brackets = max(0, brackets - 1)
            elif character == "(":
                parentheses += 1
            elif character == ")":
                parentheses = max(0, parentheses - 1)
            elif character == "," and braces == brackets == parentheses == 0:
                break
        end += 1
    if end <= start or not text[start:end].strip():
        return None
    return start, end


def extract_output_text(response: str) -> tuple[str | None, bool, bool]:
    open_match = OUTPUT_TAG.search(response)
    if not open_match:
        return None, False, bool(OUTPUT_CLOSE_TAG.search(response))
    close_match = OUTPUT_CLOSE_TAG.search(response, open_match.end())
    tail_end = close_match.start() if close_match else len(response)
    content = response[open_match.end() : tail_end].strip()
    fence = FENCE.search(content)
    if fence:
        content = fence.group(1).strip()
    return content, True, close_match is not None


def parse_re2_response(response: Any, max_repairs: int = 25) -> Re2ParseResult:
    text = str(response or "")
    extracted, has_open_tag, has_close_tag = extract_output_text(text)
    if extracted is None:
        return Re2ParseResult(
            False,
            "missing_output_tag",
            None,
            None,
            "",
            0,
            "[OUTPUT] marker not found",
            has_open_tag,
            has_close_tag,
        )
    document = re.sub(r",(\s*[}\]])", r"\1", _remove_hash_comments(extracted)).strip()
    if not document:
        return Re2ParseResult(
            False,
            "empty_output",
            None,
            None,
            document,
            0,
            "Output block is empty",
            has_open_tag,
            has_close_tag,
        )
    for repair_count in range(max_repairs + 1):
        try:
            value = json.loads(document)
            return Re2ParseResult(
                True,
                "parsed",
                "strict_json" if repair_count == 0 else "repaired_json",
                value,
                document,
                repair_count,
                None,
                has_open_tag,
                has_close_tag,
            )
        except json.JSONDecodeError as error:
            if repair_count == max_repairs:
                message = f"{error.msg} at position {error.pos}"
                break
            span = _value_span(document, error.pos)
            if span is None:
                message = f"{error.msg} at position {error.pos}"
                break
            start, end = span
            repaired = document[:start] + json.dumps(document[start:end].strip()) + document[end:]
            if repaired == document:
                message = f"{error.msg} at position {error.pos}"
                break
            document = repaired
    return Re2ParseResult(
        False,
        "invalid_json",
        None,
        None,
        document,
        repair_count,
        message,
        has_open_tag,
        has_close_tag,
    )


def preparsed_result(value: Any) -> Re2ParseResult:
    return Re2ParseResult(
        True,
        "preparsed",
        "preparsed",
        value,
        "",
        0,
        None,
        False,
        False,
    )


def unwrap_output(value: Any) -> Any:
    if isinstance(value, dict) and "output" in value:
        return value["output"]
    return value


def _looks_like_numpy_dump(value: str) -> bool:
    stripped = value.strip()
    if stripped.startswith("array(") and stripped.endswith(")"):
        return True
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return False
    return bool(
        re.search(
            r"nan|[-+]?inf|[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d*\.\d+|\d+",
            stripped,
            re.IGNORECASE,
        )
    )


def _parse_numpy_dump(value: str) -> list[Any]:
    stripped = value.strip()
    if stripped.startswith("array(") and stripped.endswith(")"):
        stripped = stripped[6:-1].strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1].strip()
    rows = re.split(r"\]\s*\[|\n", stripped)
    parsed_rows = []
    token_pattern = re.compile(
        r"nan|[-+]?inf|[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d*\.\d+|\d+", re.IGNORECASE
    )
    for row in rows:
        parsed = []
        for token in token_pattern.findall(row):
            lowered = token.lower()
            if lowered == "nan":
                parsed.append(None)
            elif lowered == "inf":
                parsed.append(math.inf)
            elif lowered == "-inf":
                parsed.append(-math.inf)
            else:
                parsed.append(float(token))
        if parsed:
            parsed_rows.append(parsed)
    return parsed_rows[0] if len(parsed_rows) == 1 else parsed_rows


def _literal_string_value(value: str) -> tuple[bool, Any]:
    stripped = value.strip()
    if stripped == "set()":
        return True, set()
    for prefix, constructor in (("set(", set), ("frozenset(", frozenset)):
        if stripped.startswith(prefix) and stripped.endswith(")"):
            try:
                inner = ast.literal_eval(stripped[len(prefix) : -1])
                return True, constructor(inner)
            except (ValueError, SyntaxError, TypeError):
                return False, value
    if stripped.startswith(("{", "[", "(")) and stripped.endswith(("}", "]", ")")):
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return False, value
        if isinstance(parsed, (dict, list, tuple, set, frozenset)):
            return True, parsed
    return False, value


def safe_compatibility_normalize(value: Any, *, parse_scalar_string: bool = False) -> Any:
    if isinstance(value, str):
        if _looks_like_numpy_dump(value):
            return safe_compatibility_normalize(_parse_numpy_dump(value))
        parsed, parsed_value = _literal_string_value(value)
        if parsed:
            return safe_compatibility_normalize(parsed_value)
        if parse_scalar_string:
            try:
                scalar = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError):
                return value
            if not isinstance(scalar, str):
                return safe_compatibility_normalize(scalar)
        # The released evaluator strips outer whitespace before string comparison.
        return value.strip()
    if isinstance(value, list):
        return [safe_compatibility_normalize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(safe_compatibility_normalize(item) for item in value)
    if isinstance(value, dict):
        return {
            key: safe_compatibility_normalize(item)
            for key, item in value.items()
        }
    if isinstance(value, set):
        return {safe_compatibility_normalize(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(safe_compatibility_normalize(item) for item in value)
    return value


def score_re2_output(
    parse: Re2ParseResult,
    target: Any,
    *,
    finish_reason: Any = "",
    native_finish_reason: Any = "",
) -> Re2Score:
    prediction = unwrap_output(parse.value) if parse.parsed else None
    typed_correct = bool(parse.parsed and recursively_typed_equal(prediction, target))
    normalized_prediction = safe_compatibility_normalize(
        prediction,
        parse_scalar_string=not isinstance(target, str),
    )
    normalized_target = safe_compatibility_normalize(target)
    compatibility_correct = bool(
        parse.parsed
        and recursively_typed_equal(normalized_prediction, normalized_target)
    )
    return Re2Score(
        strict_json_typed_correct=bool(parse.mode == "strict_json" and typed_correct),
        repaired_json_typed_correct=typed_correct,
        safe_compatibility_correct=compatibility_correct,
        parse_status=parse.status,
        parse_mode=parse.mode,
        parse_repair_count=parse.repair_count,
        prediction_type=type(prediction).__name__ if parse.parsed else None,
        target_type=type(target).__name__,
        extracted_text=parse.extracted_text,
        has_output_open_tag=parse.has_open_tag,
        has_output_close_tag=parse.has_close_tag,
        finish_reason=str(finish_reason or ""),
        native_finish_reason=str(native_finish_reason or ""),
        is_truncated=is_truncated_response(finish_reason, native_finish_reason),
    )
