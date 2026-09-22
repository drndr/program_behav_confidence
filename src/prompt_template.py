"""Prediction and verification prompt templates; no reference answers included."""

import hashlib


HEADER = "Predict the output of this Python 3 code for the given input. Do not execute it."
RE2_INPUT_NOTE = (
    "INPUT is a JSON object whose keys self, args, kwargs and input give the instance state, "
    "positional and keyword arguments, and standard input text. Strings such as \"array([...])\" "
    "or \"<class '...'>\" stand for the corresponding Python objects; if the output is such an "
    "object, give its repr() as a string.")
PREDICTION_ANSWER = ("Answer with the exact output as a Python literal inside "
                     "<prediction></prediction> tags and nothing else.")

# Verification (confidence judge) wording is not decided yet. The structure is fixed:
# the judge sees exactly what the predictor saw plus the candidate answer, never the
# reference, and answers with <verdict>correct|incorrect</verdict> (see scoring.parse_verdict).
VERIFICATION_HEADER = ""  # TODO: decide wording
VERIFICATION_ANSWER = ""  # TODO: decide wording; must ask for the <verdict> tag


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


def verification_prompt(example, candidate):
    """Judge a fixed candidate answer; empty wording slots are left out until decided."""
    context = CONTEXTS[example["dataset"]](example)
    parts = [VERIFICATION_HEADER, task_sentence(example), context,
             f"[CANDIDATE]\n{candidate}\n[/CANDIDATE]", VERIFICATION_ANSWER]
    return "\n\n".join(part for part in parts if part)


def verification_wording_final():
    return bool(VERIFICATION_HEADER and VERIFICATION_ANSWER)


def _template_sha256(render):
    """Identify wording and layout by rendering fixed probes; changes whenever either changes."""
    probes = [{"dataset": "cruxeval", "entrypoint": "f", "code": "CODE", "input": "INPUT"},
              {"dataset": "re2", "entrypoint": "g", "code": "CODE", "input": "INPUT"},
              {"dataset": "re2", "entrypoint": "", "code": "CODE", "input": "INPUT"}]
    return hashlib.sha256("\n".join(render(p) for p in probes).encode("utf-8")).hexdigest()


def prediction_template_sha256():
    return _template_sha256(prediction_prompt)


def verification_template_sha256():
    return _template_sha256(lambda example: verification_prompt(example, "CANDIDATE"))
