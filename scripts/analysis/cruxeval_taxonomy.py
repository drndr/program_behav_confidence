"""Pure error-taxonomy helpers for CRUXEval output predictions."""

import ast
import difflib
import string
import warnings
from typing import Any

# Re-exported so importers keep getting the single shared EVAL_FAIL sentinel:
# a second object() would silently break every `is EVAL_FAIL` check.
from utils import EVAL_FAIL, literal_eval_safe


SEPARATORS = set(string.punctuation + string.whitespace)


def literal_eval_type(value: str) -> tuple[Any, str | None]:
    parsed = literal_eval_safe(value)
    if parsed is not EVAL_FAIL:
        return parsed, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            ast.literal_eval((value or "").strip())
        except Exception as exc:
            return None, type(exc).__name__
    return None, "EvalFail"


def categorize_error(
    prediction: Any,
    prediction_error: str | None,
    target: Any,
    target_error: str | None,
) -> str:
    if prediction_error is not None:
        return f"unparseable_prediction:{prediction_error}"
    if target_error is not None:
        return f"unparseable_target:{target_error}"
    if type(prediction) is not type(target):
        return f"type_mismatch:{type(target).__name__}->{type(prediction).__name__}"
    if isinstance(target, (list, tuple)) and len(prediction) != len(target):
        return "sequence_length_mismatch"
    if isinstance(target, dict) and len(prediction) != len(target):
        return "dict_size_mismatch"
    return "same_type_wrong_value"


def target_type_name(target: Any, target_error: str | None) -> str:
    if target_error is not None:
        return f"unparsed_target:{target_error}"
    return type(target).__name__


def type_error_category(
    target: Any,
    target_error: str | None,
    prediction: Any,
    prediction_error: str | None,
    prediction_clean: str,
    is_correct: bool,
) -> str:
    if target_error is not None:
        return "target_parse_error"
    if is_correct:
        return "correct"
    if isinstance(target, str):
        return string_error_type(target, prediction, prediction_clean)
    if prediction_error is not None or type(prediction) is not type(target):
        return wrong_type_category(target)
    if isinstance(target, (list, tuple)):
        return sequence_error_type(target, prediction)
    if isinstance(target, dict):
        return dict_error_type(target, prediction)
    if isinstance(target, bool):
        return "flipped_bool"
    if isinstance(target, int):
        return int_error_type(target, prediction)
    if isinstance(target, float):
        return "wrong_float"
    if isinstance(target, bytes):
        return bytes_error_type(target, prediction)
    return "wrong_value"


def wrong_type_category(target: Any) -> str:
    return f"not_{type(target).__name__}"


def sequence_error_type(
    target: list[Any] | tuple[Any, ...],
    prediction: Any,
) -> str:
    if len(target) != len(prediction):
        try:
            if set(map(repr, target)) == set(map(repr, prediction)):
                return "multiplicity_or_length"
        except TypeError:
            pass
        return "length_mismatch"
    if sorted(map(repr, target)) == sorted(map(repr, prediction)):
        return "ordering_error"
    return "wrong_values_same_length"


def string_error_type(target: str, prediction: Any, prediction_clean: str) -> str:
    if prediction_clean == target:
        return "formatting_only"
    prediction_text = prediction if isinstance(prediction, str) else prediction_clean
    target_no_separators = remove_separators(target)
    prediction_no_separators = remove_separators(prediction_text)
    if prediction_no_separators == target_no_separators:
        return "separator_only_error"
    if edit_distance_at_most_one(prediction_no_separators, target_no_separators):
        return "separator_plus_one_char_error"
    if prediction_text in target:
        return "substring_of_target"
    if target in prediction_text:
        return "target_substring_of_prediction"
    if prediction_text.casefold() == target.casefold():
        return "case_only_error"
    if sorted(prediction_text) == sorted(target):
        return "character_multiset_error"
    if len(prediction_text) == len(target):
        return "same_length_wrong_chars"
    return edit_distance_error_type(target, prediction_text)


def remove_separators(value: str) -> str:
    return "".join(char for char in value if char not in SEPARATORS)


def edit_distance_at_most_one(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    left_index = 0
    edits = 0
    same_length = len(left) == len(right)
    for char in right:
        if left_index < len(left) and left[left_index] == char:
            left_index += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if same_length:
            left_index += 1
    return True


def edit_distance_error_type(target: str, prediction_clean: str) -> str:
    distance = edit_distance(target, prediction_clean)
    if distance <= 1:
        return "residual_edit_distance_1"
    if distance == 2:
        return "residual_edit_distance_2"
    if distance <= 5:
        return "residual_edit_distance_3_to_5"
    if distance <= 10:
        return "residual_edit_distance_6_to_10"
    return "residual_edit_distance_gt_10"


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + int(left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def string_residual_analysis(
    target: Any,
    prediction: Any,
    prediction_clean: str,
    input_text: str,
    type_error: str,
) -> dict[str, Any] | None:
    if not isinstance(target, str) or not type_error.startswith("residual_edit_distance_"):
        return None
    prediction_text = prediction if isinstance(prediction, str) else prediction_clean
    input_candidates = input_string_candidates(input_text)
    pred_target_distance = edit_distance(prediction_text, target)
    if not input_candidates:
        return {
            "input_relation": "no_string_input_candidate",
            "edit_shape": edit_shape(prediction_text, target),
            "target_distance_band": edit_distance_band(pred_target_distance),
            "pred_target_distance": pred_target_distance,
            "pred_input_distance": None,
            "input_target_distance": None,
            "nearest_input": None,
        }
    nearest_input = min(input_candidates, key=lambda candidate: edit_distance(prediction_text, candidate))
    pred_input_distance = edit_distance(prediction_text, nearest_input)
    input_target_distance = min(edit_distance(candidate, target) for candidate in input_candidates)
    return {
        "input_relation": input_relation(pred_target_distance, pred_input_distance, input_target_distance),
        "edit_shape": edit_shape(prediction_text, target),
        "target_distance_band": edit_distance_band(pred_target_distance),
        "pred_target_distance": pred_target_distance,
        "pred_input_distance": pred_input_distance,
        "input_target_distance": input_target_distance,
        "nearest_input": nearest_input,
    }


def input_string_candidates(input_text: str) -> list[str]:
    candidates = [str(input_text).strip()]
    parsed = literal_eval_safe(input_text)
    if parsed is not EVAL_FAIL:
        candidates.extend(string_leaves(parsed))
    return dedupe_nonempty(candidates)


def string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        leaves = []
        for item in value:
            leaves.extend(string_leaves(item))
        return leaves
    if isinstance(value, dict):
        leaves = []
        for key, item in value.items():
            leaves.extend(string_leaves(key))
            leaves.extend(string_leaves(item))
        return leaves
    return []


def dedupe_nonempty(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def input_relation(pred_target_distance: int, pred_input_distance: int, input_target_distance: int) -> str:
    if pred_input_distance == 0:
        return "input_echo_exact"
    if pred_input_distance <= 1:
        return "input_echo_near"
    if input_target_distance == 0:
        return "target_also_in_input_wrong"
    if pred_target_distance < input_target_distance and pred_input_distance < input_target_distance:
        return "between_input_and_target"
    if pred_input_distance < pred_target_distance:
        return "closer_to_input"
    if pred_target_distance < pred_input_distance:
        return "closer_to_target"
    return "far_from_input_and_target"


def edit_shape(prediction: str, target: str) -> str:
    regions = sum(1 for opcode in difflib.SequenceMatcher(None, prediction, target).get_opcodes() if opcode[0] != "equal")
    if regions <= 1:
        return "one_edit_region"
    if regions == 2:
        return "two_edit_regions"
    return "three_plus_edit_regions"


def edit_distance_band(distance: int) -> str:
    if distance <= 2:
        return "<=2"
    if distance <= 5:
        return "3-5"
    if distance <= 10:
        return "6-10"
    return ">10"


def dict_error_type(
    target: dict[Any, Any],
    prediction: Any,
) -> str:
    if set(prediction) != set(target):
        return "key_mismatch"
    return "value_mismatch"


def int_error_type(target: int, prediction: Any) -> str:
    if prediction == -target:
        return "sign_error"
    if abs(prediction - target) == 1:
        return "off_by_one"
    return "wrong_int"


def bytes_error_type(target: bytes, prediction: Any) -> str:
    if len(prediction) != len(target):
        return "length_mismatch"
    return "wrong_bytes"
