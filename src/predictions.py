"""Prompt protocol and scoring prototype. No model calls or program execution.

Two prompts per item. The prediction prompt asks for the output as a Python
literal inside <prediction> tags. The confidence prompt shows a fixed candidate
answer and asks for a binary verdict inside <verdict> tags. CRUXEval and RE2
share one skeleton; RE2 adds one paragraph that explains its JSON input format.
No worked examples, no reasoning instructions, and never the reference answer.
"""

import ast
import csv
import hashlib
import json
from pathlib import Path
import re


HEADER = "Predict the output of this Python 3 code for the given input. Do not execute it."
JUDGE_HEADER = ("A candidate answer was given for the output of this Python 3 code. "
                "Judge it without executing the code.")
RE2_INPUT_NOTE = (
    "INPUT is a JSON object whose keys self, args, kwargs and input give the instance state, "
    "positional and keyword arguments, and standard input text. Strings such as \"array([...])\" "
    "or \"<class '...'>\" stand for the corresponding Python objects; if the output is such an "
    "object, give its repr() as a string.")
PREDICTION_ANSWER = ("Answer with the exact output as a Python literal inside "
                     "<prediction></prediction> tags and nothing else.")
VERDICT_ANSWER = ("Is the candidate exactly correct? Answer correct or incorrect inside "
                  "<verdict></verdict> tags and nothing else.")


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


# --- prompts ------------------------------------------------------------------

def task_sentence(example):
    if example["entrypoint"]:
        return f"Predict the value returned by `{example['entrypoint']}` when called with INPUT."
    return "Predict what the program prints when INPUT is given on standard input."


def cruxeval_context(example):
    return f"[CODE]\n{example['code']}\n[/CODE]\n\n[INPUT]\n{example['input']}\n[/INPUT]"


def re2_context(example):
    return cruxeval_context(example) + "\n\n" + RE2_INPUT_NOTE


CONTEXTS = {"cruxeval": cruxeval_context, "re2": re2_context}


def prediction_prompt(example):
    context = CONTEXTS[example["dataset"]](example)
    return f"{HEADER}\n\n{task_sentence(example)}\n\n{context}\n\n{PREDICTION_ANSWER}"


def confidence_prompt(example, prediction):
    """Judge a fixed candidate answer; the judge sees what the predictor saw plus the candidate."""
    context = CONTEXTS[example["dataset"]](example)
    return (f"{JUDGE_HEADER}\n\nThe task was: {task_sentence(example)}\n\n{context}\n\n"
            f"[CANDIDATE]\n{prediction}\n[/CANDIDATE]\n\n{VERDICT_ANSWER}")


def template_sha256():
    """Identifies the prompt wording and layout; changes whenever either changes."""
    probes = [{"dataset": "cruxeval", "entrypoint": "f", "code": "CODE", "input": "INPUT"},
              {"dataset": "re2", "entrypoint": "g", "code": "CODE", "input": "INPUT"},
              {"dataset": "re2", "entrypoint": "", "code": "CODE", "input": "INPUT"}]
    text = "\n".join(prediction_prompt(p) + "\n" + confidence_prompt(p, "CANDIDATE") for p in probes)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        return bool(re.search(r"\b(?:array|masked_array|np\.[A-Za-z_]+)\b|<[^>]+>", value))
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
    return left == right


def score_prediction(example, text):
    """Score one raw model response against the example's reference."""
    try:
        literal = parse_prediction(text)
    except ValueError as error:
        return {"status": "invalid_response", "correct": None, "detail": str(error)}
    try:
        reference = reference_value(example)
    except (ValueError, SyntaxError):
        return {"status": "unsupported_reference", "correct": None}
    try:
        prediction = prediction_value(example, literal)
    except (ValueError, SyntaxError, TypeError):
        return {"status": "invalid_prediction", "correct": None}
    if example["dataset"] == "re2" and representation_pending(reference):
        return {"status": "needs_scoring_policy", "correct": None, "exact_match": same_value(prediction, reference)}
    return {"status": "scored", "correct": same_value(prediction, reference)}
