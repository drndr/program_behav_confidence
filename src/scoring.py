"""Response parsing and scoring. No model calls, no execution of model text.

Three nested scoring modes. Every completed response is scored right or wrong in
every mode; the modes differ only in how much is normalized before comparing.

  strict      typed Python literal equality; RE2 predictions pass through json.dumps/loads
              the way upstream produced its references (tuples -> lists, dict keys -> strings)
  relaxed     strict, plus an unquoted answer compared as string content
  normalized  relaxed, plus RE2 representation rules keyed on the reference text: numeric
              or boolean array displays (formatting and dtype ignored), prettyForm displays
              (content only), literals JSON could not store (complex, tuple, bytes), and any
              other object repr or expression compared as a syntax tree where positional
              arguments are content and must match, while keywords the prediction adds beyond
              the reference are taken as spelled-out defaults (never for Python builtins and
              standard collections, whose keywords are content); applied inside lists and dicts

Only ast parsing and literal evaluation are used; a prediction is never executed.
"""

import ast
import builtins
import collections
import json
import math
import re

SCORING_MODES = ("strict", "relaxed", "normalized")


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


def prediction_logprobs(text, logprobs):
    """Mean log-probability of the tokens that produced the <prediction> content.

    Tokens are concatenated by their bytes and the prediction content is located in
    that stream between its tags (providers may prepend a </think> token, append an
    end marker, or drop a tag's first token). A token overlapping the content counts,
    so tag-adjacent tokens such as ">'" are included. Returns None when the stream
    does not reproduce the content exactly.
    """
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    match = re.search(r"<prediction>(.*?)</prediction>", text, re.S)
    if not content or match is None:
        return None
    inner = match.group(1)
    lead, trail = len(inner) - len(inner.lstrip()), len(inner) - len(inner.rstrip())
    if lead + trail >= len(inner):
        return None
    spans, offset, pieces = [], 0, []
    for token in content:
        raw = bytes(token["bytes"]) if token.get("bytes") is not None else str(token.get("token") or "").encode("utf-8")
        if not isinstance(token.get("logprob"), (int, float)):
            return None
        spans.append((offset, offset + len(raw), float(token["logprob"])))
        pieces.append(raw)
        offset += len(raw)
    stream = b"".join(pieces)
    # Anchor on tag suffixes: providers sometimes drop the first token of a tag ("<p", "</")
    # from the stream, but the content itself must be reproduced exactly.
    pattern = rb"rediction>(" + re.escape(inner.encode("utf-8")) + rb")(?:</)?prediction>"
    found = list(re.finditer(pattern, stream, re.S))
    if not found:
        return None
    located = found[-1]
    content_start = located.start(1) + len(inner[:lead].encode("utf-8"))
    content_end = located.end(1) - len(inner[len(inner) - trail:].encode("utf-8"))
    selected = [logprob for start, end, logprob in spans if start < content_end and end > content_start]
    if not selected:
        return None
    mean = sum(selected) / len(selected)
    return {"token_count": len(selected), "sum_logprob": sum(selected), "mean_logprob": mean,
            "geometric_mean_probability": math.exp(mean)}


# --- literals -----------------------------------------------------------------

_NON_FINITE = {"inf": math.inf, "infinity": math.inf, "nan": math.nan}


class _NonFinite(ast.NodeTransformer):
    """Rewrite inf, nan and float('inf'|'-inf'|'nan') into constants; nothing else."""

    def visit_Name(self, node):
        if node.id in _NON_FINITE:
            return ast.copy_location(ast.Constant(_NON_FINITE[node.id]), node)
        return node

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Name) and node.func.id == "float" and len(node.args) == 1
                and not node.keywords and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.strip().lower().lstrip("+-") in _NON_FINITE):
            return ast.copy_location(ast.Constant(float(node.args[0].value)), node)
        return self.generic_visit(node)


def parse_literal(text):
    """Parse literal text into an AST expression, accepting inf/nan spellings."""
    tree = ast.parse(text.strip(), mode="eval")
    return ast.fix_missing_locations(_NonFinite().visit(tree)).body


def literal_value(node):
    return ast.literal_eval(node)


def reference_value(example):
    if example["dataset"] == "cruxeval":
        return literal_value(parse_literal(example["output"]))
    return json.loads(example["output"])


def prediction_value(example, literal):
    """Evaluate a predicted literal the way the dataset's references were produced."""
    value = literal_value(parse_literal(literal))
    if example["dataset"] == "re2":
        # Upstream RE2 recorded json.dumps(return value): tuples become lists, dict keys strings.
        value = json.loads(json.dumps(value))
    return value


def same_value(left, right):
    """Exact values and types: no bool/number, tuple/list, or float coercion. NaN equals NaN."""
    if type(left) is not type(right):
        return False
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return len(left) == len(right) and all(
            any(same_value(k, other) and same_value(v, right[other]) for other in right)
            for k, v in left.items())
    if isinstance(left, (set, frozenset)):
        return len(left) == len(right) and all(any(same_value(a, b) for b in right) for a in left)
    return left == right


# --- example inspection --------------------------------------------------------

def representation_pending(value):
    """Reference text that stands for a library object; used for example selection only."""
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


# --- normalized representation rules ---------------------------------------------

_RULE_ERRORS = (ValueError, SyntaxError, TypeError, AttributeError, RecursionError)


def _array_values(text):
    """Read numeric/boolean array displays without evaluating calls; dtype is ignored."""
    text = text.strip()
    if "..." in text:
        raise ValueError("Abbreviated array display")
    if text.startswith("array(") or text.startswith("np.array("):
        node = ast.parse(text, mode="eval").body
        if not isinstance(node, ast.Call) or len(node.args) != 1 or any(
                k.arg != "dtype" for k in node.keywords):
            raise ValueError("Unsupported array representation")
        return ast.literal_eval(node.args[0])
    # NumPy str(array) omits commas. Only accept brackets, numbers and booleans.
    if not re.fullmatch(r"(?:[\[\]\s,0-9eE+.\-]|True|False)+", text):
        raise ValueError("Unsupported array representation")
    text = re.sub(r"(?<=[0-9.\]e])\s+(?=[+\-0-9.\[TF])", ",", text)
    return ast.literal_eval(text)


def _numeric_equal(left, right):
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_numeric_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
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


_JSON_TYPES = (str, int, float, bool, type(None))
_NOT_A_LITERAL = object()
# Python's own constructors take content, never parameters at defaults: dict(a=1) is not dict().
_CONTENT_CALLABLES = frozenset(name for name in dir(builtins) if not name.startswith("_")) | frozenset(
    name for name in dir(collections) if not name.startswith("_")) | {"SimpleNamespace"}


def _tree_equal(left, right):
    """Structural equality of expression trees.

    In a call, positional arguments are content and must match. Keywords in the
    left (reference) tree must match; keywords only the right tree has are taken
    as parameters spelled out at their defaults, except for Python builtins and
    standard collections, whose keywords are content and must match exactly.
    """
    if isinstance(left, ast.Call) and isinstance(right, ast.Call):
        callee = ast.unparse(left.func)
        if callee != ast.unparse(right.func) or len(left.args) != len(right.args):
            return False
        if not all(_tree_equal(a, b) for a, b in zip(left.args, right.args)):
            return False
        if any(k.arg is None for k in left.keywords + right.keywords):
            return False  # ** unpacking: contents unknown
        given = {k.arg: k.value for k in right.keywords}
        if callee.split(".")[-1] in _CONTENT_CALLABLES and set(given) != {k.arg for k in left.keywords}:
            return False
        return all(k.arg in given and _tree_equal(k.value, given[k.arg]) for k in left.keywords)
    if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
        return type(left.value) is type(right.value) and left.value == right.value
    if type(left) is not type(right):
        return False
    if isinstance(left, ast.AST):
        return all(_tree_equal(getattr(left, field), getattr(right, field))
                   for field in left._fields if field != "ctx")
    if isinstance(left, list):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def _representation_rule(reference, source):
    """Compare a candidate's text with an object-like reference string; raises when no rule applies."""
    reference = reference.strip()
    if re.match(r"^prettyForm\s*\(", reference):
        return _pretty_content(reference) == _pretty_content(source)
    if reference.startswith(("array(", "np.array(")) or (reference.startswith("[") and reference.endswith("]")):
        return _numeric_equal(_array_values(reference), _array_values(source))
    expression = parse_literal(reference)  # SyntaxError: the reference is display text, not an expression.
    try:
        value = literal_value(expression)
    except _RULE_ERRORS:
        value = _NOT_A_LITERAL
    if value is not _NOT_A_LITERAL:
        if isinstance(value, _JSON_TYPES):
            # A plain value stored as text (e.g. "4"): strict already decided; keep the type distinction.
            raise ValueError("Plain value stored as text")
        # A literal JSON could not store (complex, tuple, bytes): compare values.
        return same_value(value, literal_value(parse_literal(source)))
    # Object reprs and expressions: positional arguments are content, extra keywords are defaults.
    return _tree_equal(expression, parse_literal(source))


def _normalized_equal(reference, node, source_text, bare_text=None):
    """Reference-guided comparison with representation rules; always True or False."""
    if isinstance(reference, str):
        try:
            candidate = literal_value(node)
        except _RULE_ERRORS:
            candidate = bare_text if bare_text is not None else ast.get_source_segment(source_text, node)
        if candidate == reference:
            return True
        source = candidate if isinstance(candidate, str) else ast.get_source_segment(source_text, node)
        if source is None:
            return False
        try:
            return bool(_representation_rule(reference, source))
        except _RULE_ERRORS:
            return False
    if isinstance(reference, list) and isinstance(node, (ast.List, ast.Tuple)):
        return len(reference) == len(node.elts) and all(
            _normalized_equal(r, n, source_text) for r, n in zip(reference, node.elts))
    if isinstance(reference, dict) and isinstance(node, ast.Dict):
        try:
            keys = [next(iter(json.loads(json.dumps({literal_value(k): 0})))) for k in node.keys]
        except _RULE_ERRORS:
            return False
        if len(keys) != len(set(keys)) or set(keys) != set(reference):
            return False
        return all(_normalized_equal(reference[k], n, source_text) for k, n in zip(keys, node.values))
    try:
        return same_value(literal_value(node), reference)
    except _RULE_ERRORS:
        return False


# --- scoring ------------------------------------------------------------------

def score_prediction(example, text, mode="strict"):
    """Score one raw model response against the example's reference.

    Returns status and correct. correct is True or False for every response that
    contains a prediction; status records why a response could not be read as a
    literal (invalid_response: no single tag; invalid_prediction: not a literal).
    """
    if mode not in SCORING_MODES:
        raise ValueError(f"Unknown scoring mode: {mode}")
    try:
        literal = parse_prediction(text)
    except ValueError as error:
        return {"status": "invalid_response", "correct": False, "detail": str(error)}
    try:
        reference = reference_value(example)
    except (ValueError, SyntaxError):
        return {"status": "unsupported_reference", "correct": None}
    try:
        prediction = prediction_value(example, literal)
        parsed = True
    except _RULE_ERRORS:
        parsed = False
    if parsed and same_value(prediction, reference):
        return {"status": "scored", "correct": True}
    wrong = {"status": "scored" if parsed else "invalid_prediction", "correct": False}
    if mode == "strict":
        return wrong
    # Keep spaces: they may be the actual output. Surrounding line breaks are formatting.
    bare = re.findall(r"<prediction>(.*?)</prediction>", text, re.S)[0].strip("\r\n")
    if not parsed and isinstance(reference, str) and bare == reference:
        return {"status": "scored", "correct": True, "detail": "unquoted string content"}
    if mode == "relaxed" or example["dataset"] != "re2":
        return wrong
    try:
        node = parse_literal(literal)
    except SyntaxError:
        node = None  # e.g. NumPy's comma-less display; try the rules on the raw text.
    try:
        matched = (_normalized_equal(reference, node, literal, bare) if node is not None
                   else isinstance(reference, str) and _representation_rule(reference, bare))
    except _RULE_ERRORS:
        matched = False
    if matched:
        return {"status": "scored", "correct": True, "detail": "representation rule"}
    return wrong
