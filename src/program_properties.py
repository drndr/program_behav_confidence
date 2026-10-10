"""Reusable program properties: static size and control-flow measurements.

Executable lines here means physical source lines with code tokens (including
definition/control headers and continuation lines), excluding blank/comment
lines and standalone string-literal statements such as docstrings. This is a
source-size convention, not a count of bytecode instructions or executed lines.

Cyclomatic complexity (control flow) and Halstead volume and difficulty (operator and
operand vocabulary) are measured on the whole supplied code and on the target function
alone (the entry point the model is asked about); information flow (what the target
reads from and sends outside itself) on the target only.
"""

import ast
import io
import keyword
import math
import re
import tokenize


# Tokens that are neither Halstead operators nor operands: layout, comments, f-string delimiters.
HALSTEAD_SKIPPED = {tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT,
                    tokenize.ENCODING, tokenize.ENDMARKER, tokenize.FSTRING_START, tokenize.FSTRING_END}

# RE2 names each helper on an unindented line: a dotted path, sometimes dot-prefixed or with a
# hyphenated package ("scikit-learn.sklearn..."), or a comprehension/lambda frame ("pkg.mod.<genexpr>").
HELPER_LABEL = re.compile(r"\.?(?:[A-Za-z_][\w-]*\.)+(?:[A-Za-z_]\w*|<(?:genexpr|listcomp|dictcomp|setcomp|lambda)>)")


def helper_label_lines(code):
    """Line numbers of RE2 helper labels, which are not Python.

    A dotted path is a label only before a definition, since it could otherwise be code.
    A frame name is never Python; it precedes a code fragment, which is kept.
    """
    lines = code.splitlines()
    excluded = set()
    for i, line in enumerate(lines):
        if not HELPER_LABEL.fullmatch(line):
            continue
        following = next((s for s in lines[i + 1:] if s.strip()), "").lstrip()
        if line.endswith(">") or re.match(r"(?:async\s+def\s|def\s|class\s|@)", following):
            excluded.add(i + 1)
    return excluded


def without_helper_labels(code, dataset):
    """The code with RE2 helper labels blanked (line numbers kept), and the label line numbers."""
    labels = helper_label_lines(code) if dataset == "re2" else set()
    lines = code.splitlines(keepends=True)
    return "".join("\n" if i in labels else line for i, line in enumerate(lines, 1)), labels


def executable_lines(code, dataset):
    """Return count and diagnostics; incomplete tokenization produces no count."""
    source, labels = without_helper_labels(code, dataset)
    lines = code.splitlines(keepends=True)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        if any(t.type == tokenize.ERRORTOKEN and not t.string.isspace() for t in tokens):
            raise ValueError("Unrecognized Python token")
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError) as error:
        return {"executable_lines": None, "line_status": "tokenization_failed",
                "line_detail": str(error), "helper_labels_excluded": len(labels)}
    counted, statement = set(), []
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}

    def flush():
        # Exclude only standalone literal strings, not assigned strings or f-strings.
        is_doc = any(t.type == tokenize.STRING for t in statement) and all(
            (t.type == tokenize.STRING and not t.string.lower().startswith(("f", "b", "rf", "fr", "rb", "br")))
            or (t.type == tokenize.OP and t.string in {"(", ")"}) for t in statement)
        if not is_doc:
            for token in statement:
                for lineno in range(token.start[0], token.end[0] + 1):
                    if lineno <= len(lines) and lines[lineno - 1].strip():
                        counted.add(lineno)
        statement.clear()

    for token in tokens:
        if token.type in {tokenize.NEWLINE, tokenize.ENDMARKER}:
            flush()
        elif token.type not in ignored:
            statement.append(token)
    try:
        ast.parse(source)
        status, detail = "ok", ""
    except (SyntaxError, ValueError) as error:
        # Lexical source size remains defined, but this is not validated Python.
        status, detail = "lexical_only", str(error)
    return {"executable_lines": len(counted), "line_status": status,
            "line_detail": detail, "helper_labels_excluded": len(labels)}


def cyclomatic_complexity(node):
    """McCabe complexity of an AST node: 1 + its decision points.

    Decision points: each if/elif and conditional expression, each loop (for, while, and the
    for-clauses of comprehensions), each comprehension condition, each and/or, each except
    clause and each match case. Else-branches, with, assert and return do not count.
    """
    points = 0
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While,
                              ast.ExceptHandler, ast.match_case)):
            points += 1
        elif isinstance(child, ast.comprehension):
            points += 1 + len(child.ifs)
        elif isinstance(child, ast.BoolOp):
            points += len(child.values) - 1
    return 1 + points


def target_definition(tree, entrypoint):
    """The entry point's definition and its name, or (None, "") when it cannot be told apart.

    Code without an entry point (RE2's Avatar scripts) is its own target. When several
    functions share the name, the target is the only one defined in a class: RE2 lists
    helper methods as top-level copies, while the target keeps its class.
    """
    if not entrypoint:
        return tree, "<module>"
    owners = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    found = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entrypoint]
    if len(found) > 1:
        found = [node for node in found if isinstance(owners[node], ast.ClassDef)]
    if len(found) != 1:
        return None, ""
    owner = owners[found[0]]
    return found[0], f"{owner.name}.{entrypoint}" if isinstance(owner, ast.ClassDef) else entrypoint


def halstead(tokens):
    """Halstead volume and difficulty of Python tokens, or (None, None) without operands.

    Operators are operator and punctuation tokens and keywords; operands are names, numbers,
    strings and f-string text (a docstring is one operand). Volume is the token count times
    log2 of the vocabulary; difficulty is (distinct operators / 2) * (operands / distinct operands).
    """
    operators, operands = [], []
    for token in tokens:
        if token.type in HALSTEAD_SKIPPED:
            continue
        if token.type == tokenize.OP or (token.type == tokenize.NAME and keyword.iskeyword(token.string)):
            operators.append(token.string)
        else:
            operands.append(token.string)
    distinct_operators, distinct_operands = len(set(operators)), len(set(operands))
    if not distinct_operands:
        return None, None
    volume = (len(operators) + len(operands)) * math.log2(distinct_operators + distinct_operands)
    return round(volume, 2), round(distinct_operators / 2 * len(operands) / distinct_operands, 2)


def information_flow(target):
    """Henry-Kafura information flow without the length term: (fan-in * fan-out) ** 2.

    Fan-in: parameters (other than self), names read from outside the target, and instance
    attributes (self.x) read. Fan-out: distinct functions called (a method by its name) and
    instance attributes written. Calls into the target are not part of the supplied code, so
    fan-in counts data only. A target that calls and writes nothing has information flow 0.
    """
    called = {node.func for node in ast.walk(target) if isinstance(node, ast.Call)}
    callees = {f.attr if isinstance(f, ast.Attribute) else ast.unparse(f) for f in called}
    parameters = set()
    if isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = target.args
        parameters = {a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs
                      + [arguments.vararg, arguments.kwarg] if a} - {"self"}
    stored = {n.id for n in ast.walk(target) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)}
    outside = {n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
               and n not in called and n.id not in stored | parameters | {"self"}}

    def attributes(context):
        return {n.attr for n in ast.walk(target) if isinstance(n, ast.Attribute) and isinstance(n.ctx, context)
                and isinstance(n.value, ast.Name) and n.value.id == "self" and n not in called}

    fan_in = len(parameters) + len(outside) + len(attributes(ast.Load))
    fan_out = len(callees) + len(attributes(ast.Store))
    return (fan_in * fan_out) ** 2


COMPLEXITY_FIELDS = ("cyclomatic_complexity", "target_cyclomatic_complexity", "halstead_volume",
                     "halstead_difficulty", "target_halstead_volume", "target_halstead_difficulty", "information_flow")


def measure_complexity(example):
    """Cyclomatic complexity and Halstead measures of the whole code and of the target.

    The target's Halstead tokens are those on its lines. Code that does not parse gets none.
    """
    source = without_helper_labels(example["code"], example["dataset"])[0]
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {**dict.fromkeys(COMPLEXITY_FIELDS), "complexity_target": ""}
    target, name = target_definition(tree, example["entrypoint"])
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    volume, difficulty = halstead(tokens)
    if target is None:
        target_volume = target_difficulty = None
    else:
        target_volume, target_difficulty = halstead(
            tokens if target is tree else [t for t in tokens if target.lineno <= t.start[0] <= target.end_lineno])
    return {"cyclomatic_complexity": cyclomatic_complexity(tree),
            "target_cyclomatic_complexity": cyclomatic_complexity(target) if target else None,
            "halstead_volume": volume, "halstead_difficulty": difficulty,
            "target_halstead_volume": target_volume, "target_halstead_difficulty": target_difficulty,
            "information_flow": information_flow(target) if target else None,
            "complexity_target": name}


def measure_size(example, encoding):
    """Tokenize the exact CSV code/input fields; no prompt wrapper or target."""
    return {"code_token_length": len(encoding.encode_ordinary(example["code"])),
            "input_token_length": len(encoding.encode_ordinary(example["input"])),
            **executable_lines(example["code"], example["dataset"])}


def measure_properties(example, encoding):
    """All property groups for one example, in CSV column order."""
    return {**measure_size(example, encoding), **measure_complexity(example)}
