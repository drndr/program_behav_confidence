"""Dataset loading, response parsing, and scoring prototype. No model calls."""

import ast
import csv
import json
import importlib
from pathlib import Path
import re
from functools import lru_cache

SCORING_MODES = ("strict", "relaxed", "normalized")


@lru_cache(maxsize=1)
def estimator_names():
    """Discover trusted installed classes, never import a model-supplied name."""
    from sklearn.base import BaseEstimator
    from sklearn.utils.discovery import all_estimators
    names = set()
    for name, cls in all_estimators():
        if issubclass(cls, BaseEstimator):
            names.add(name)
            names.add(f"{cls.__module__}.{name}")
            # Public sklearn modules re-export their estimator classes.
            public_module = ".".join(cls.__module__.split(".")[:2])
            if getattr(importlib.import_module(public_module), name, None) is cls:
                names.add(f"{public_module}.{name}")
    return frozenset(names)


def load_examples(path, dataset):
    """Rows with id, code, entrypoint, input, output and dataset.

    CRUXEval keeps its upstream columns; every one of its programs is a single
    function f, so the entrypoint is filled in here. RE2 CSVs carry the column.
    """
    csv.field_size_limit(10_000_000)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [dict(row, dataset=dataset) for row in csv.DictReader(stream)]
    if dataset == "cruxeval":
        rows = [dict(row, entrypoint="f") for row in rows]
    if any(row.get("entrypoint") is None for row in rows):
        raise ValueError(f"Missing entrypoint column in {path}")
    return rows


# --- responses ----------------------------------------------------------------

def _single_tag(text, tag):
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one <{tag}> block, found {len(matches)}")
    return matches[0].strip()


def parse_prediction(text):
    """The literal text inside the single <prediction> block, not evaluated."""
    return _single_tag(text, "prediction")


def parse_verdict(text):
    verdict = _single_tag(text, "verdict").lower()
    if verdict not in {"correct", "incorrect"}:
        raise ValueError(f"Verdict must be correct or incorrect, got {verdict!r}")
    return verdict


# --- scoring ------------------------------------------------------------------

def reference_value(example):
    if example["dataset"] == "cruxeval":
        return ast.literal_eval(example["output"])
    return json.loads(example["output"])


def prediction_value(example, literal):
    """Evaluate a predicted literal the way the dataset's references were produced."""
    value = ast.literal_eval(literal)
    if example["dataset"] == "re2":
        # Upstream RE2 recorded json.dumps(return value): tuples become lists, dict keys strings.
        value = json.loads(json.dumps(value, allow_nan=False))
    return value


def representation_pending(value):
    # Serialized library objects require a separate, agreed comparison policy.
    if isinstance(value, str):
        return bool(re.search(r"\b(?:array|masked_array|np\.[A-Za-z_]+)\b|<[^>]+>", value)
                    or re.match(r"^[\w.]+\s*\(", value.strip())
                    or (value.strip().startswith("[") and value.strip().endswith("]")))
    if isinstance(value, (list, tuple)):
        return any(representation_pending(v) for v in value)
    if isinstance(value, dict):
        return any(representation_pending(v) for v in value.values())
    return False


def category(example):
    try:
        value = reference_value(example)
    except (ValueError, SyntaxError):
        return "unsupported_reference"
    if example["dataset"] == "re2" and representation_pending(value):
        return "serialized_object"
    if not example["entrypoint"]:
        return "stdin/" + type(value).__name__
    return type(value).__name__


def select_examples(examples, count=8):
    """Take one per output category, then fill in source order. Not a study sample."""
    selected, seen = [], set()
    for row in examples:
        label = category(row)
        if label not in seen and len(selected) < count:
            selected.append(row)
            seen.add(label)
    ids = {r["id"] for r in selected}
    for row in examples:
        if len(selected) >= count:
            break
        if row["id"] not in ids:
            selected.append(row)
            ids.add(row["id"])
    return selected


def same_value(left, right):
    """Exact values and types: no bool/number, tuple/list, or float coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return len(left) == len(right) and all(
            any(same_value(k, other) and same_value(v, right[other]) for other in right)
            for k, v in left.items())
    if isinstance(left, (set, frozenset)):
        return len(left) == len(right) and all(any(same_value(a, b) for b in right) for a in left)
    return left == right


def _array_values(text):
    """Read numeric array displays without evaluating calls; dtype is ignored."""
    text = text.strip()
    if "..." in text:
        raise ValueError("Abbreviated array display")
    if text.startswith("array(") or text.startswith("np.array("):
        node = ast.parse(text, mode="eval").body
        if not isinstance(node, ast.Call) or len(node.args) != 1 or any(
                k.arg != "dtype" for k in node.keywords):
            raise ValueError("Unsupported array representation")
        return ast.literal_eval(node.args[0])
    # NumPy str(array) omits commas. Only accept brackets and numeric tokens.
    if not re.fullmatch(r"[\[\]\s,0-9eE+.\-]+", text):
        raise ValueError("Unsupported array representation")
    text = re.sub(r"(?<=[0-9.\]])\s+(?=[+\-0-9.\[])", ",", text)
    return ast.literal_eval(text)


def _numeric_equal(left, right):
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_numeric_equal(a, b) for a, b in zip(left, right))
    if type(left) not in (int, float, complex) or type(right) not in (int, float, complex):
        raise ValueError("Array contains unsupported values")
    return left == right  # Exact numeric equality; no tolerance or dtype comparison.


def _pretty_content(text):
    """Unwrap a prettyForm display; ignore baseline/binding, never execute it."""
    if re.match(r"^prettyForm\s*\(", text.strip()):
        call = ast.parse(text, mode="eval").body
        if not isinstance(call, ast.Call) or not call.args:
            raise ValueError("Missing prettyForm display")
        text = ast.literal_eval(call.args[0])
        if not isinstance(text, str):
            raise ValueError("Non-string prettyForm display")
    # For set displays, compare elements independent of order and string quotes.
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return ("text", text)
    if isinstance(node, ast.Set):
        elements = []
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                elements.append(element.value)
            elif isinstance(element, (ast.Name, ast.Attribute)):
                elements.append(ast.unparse(element))
            else:
                return ("text", text)
        return ("set_display", frozenset(elements))
    return ("text", text)


def _normalized_equal(reference, node, source_text, bare_text=None):
    """Reference-guided comparison: return None for unsupported object semantics."""
    if isinstance(reference, str):
        try:
            candidate = ast.literal_eval(node)
        except (ValueError, SyntaxError):
            candidate = bare_text if bare_text is not None else ast.get_source_segment(source_text, node)
        source = candidate if isinstance(candidate, str) else ast.get_source_segment(source_text, node)
        if isinstance(candidate, str) and candidate == reference:
            return True
        if re.match(r"^prettyForm\s*\(", reference.strip()):
            try:
                return _pretty_content(reference) == _pretty_content(source)
            except (ValueError, SyntaxError, TypeError):
                return None
        # Lenient class-name-only comparison; arguments are deliberately ignored.
        empty_constructor = re.fullmatch(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\(\s*\)", reference.strip())
        if empty_constructor and empty_constructor[1] in estimator_names():
            try:
                call = ast.parse(source, mode="eval").body
                return isinstance(call, ast.Call) and ast.unparse(call.func) == empty_constructor[1]
            except (ValueError, SyntaxError):
                return False
        is_array = reference.strip().startswith(("array(", "np.array(")) or (
            reference.strip().startswith("[") and reference.strip().endswith("]"))
        if is_array:
            try:
                return _numeric_equal(_array_values(reference), _array_values(source))
            except (ValueError, SyntaxError, TypeError):
                return None
        try:
            value = ast.literal_eval(reference)
            if isinstance(value, complex):
                other = ast.literal_eval(source)
                return type(other) in (int, float, complex) and value == other
        except (ValueError, SyntaxError):
            pass
        # Match expression syntax only, never algebraic equivalence or execution.
        try:
            expression = ast.parse(reference, mode="eval").body
            symbolic = isinstance(expression, ast.BinOp) or (
                isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
                and expression.func.id in {"sin", "cos", "tan", "exp", "log", "sqrt"})
            if symbolic:
                return ast.dump(expression) == ast.dump(ast.parse(source, mode="eval").body)
        except (ValueError, SyntaxError):
            pass
        if representation_pending(reference):
            return None
        return same_value(candidate, reference)
    if isinstance(reference, list) and isinstance(node, (ast.List, ast.Tuple)):
        if len(reference) != len(node.elts):
            return False
        matches = [_normalized_equal(r, n, source_text) for r, n in zip(reference, node.elts)]
        return False if False in matches else (None if None in matches else True)
    if isinstance(reference, dict) and isinstance(node, ast.Dict):
        try:
            keys = [next(iter(json.loads(json.dumps({ast.literal_eval(k): 0})))) for k in node.keys]
        except (ValueError, SyntaxError, TypeError):
            return None
        if len(keys) != len(set(keys)) or set(keys) != set(reference):
            return False
        matches = [_normalized_equal(reference[k], n, source_text) for k, n in zip(keys, node.values)]
        return False if False in matches else (None if None in matches else True)
    return same_value(ast.literal_eval(node), reference)


def score_prediction(example, text, mode="strict"):
    """strict: typed literals; relaxed: also allow unquoted, non-literal strings.

    normalized additionally compares supported RE2 object representations.
    All require one prediction tag. No numeric tolerance or object evaluation.
    """
    if mode not in SCORING_MODES:
        raise ValueError(f"Unknown scoring mode: {mode}")
    try:
        literal = parse_prediction(text)
    except ValueError as error:
        return {"status": "invalid_response", "correct": None, "detail": str(error)}
    try:
        reference = reference_value(example)
    except (ValueError, SyntaxError):
        return {"status": "unsupported_reference", "correct": None}
    if mode == "normalized" and example["dataset"] == "re2":
        try:
            bare_text = re.findall(r"<prediction>(.*?)</prediction>", text, re.S)[0].strip("\r\n")
            result = _normalized_equal(reference, ast.parse(literal, mode="eval").body, literal, bare_text)
            return {"status": "needs_scoring_policy" if result is None else "scored", "correct": result}
        except (ValueError, SyntaxError, TypeError):
            pass
    try:
        prediction = prediction_value(example, literal)
    except (ValueError, SyntaxError, TypeError):
        if example["dataset"] == "re2" and representation_pending(reference):
            return {"status": "needs_scoring_policy", "correct": None}
        if mode in {"relaxed", "normalized"} and isinstance(reference, str):
            # Keep whitespace: spaces and newlines may be the actual output.
            prediction = re.findall(r"<prediction>(.*?)</prediction>", text, re.S)[0].strip("\r\n")
            return {"status": "scored", "correct": prediction == reference,
                    "detail": "Compared unquoted string content"}
        return {"status": "invalid_prediction", "correct": None}
    if same_value(prediction, reference):
        return {"status": "scored", "correct": True}
    if example["dataset"] == "re2" and representation_pending(reference):
        return {"status": "needs_scoring_policy", "correct": None, "exact_match": same_value(prediction, reference)}
    return {"status": "scored", "correct": same_value(prediction, reference)}
