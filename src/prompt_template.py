"""Prediction and verification prompt templates; no reference answers included."""

import hashlib


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
