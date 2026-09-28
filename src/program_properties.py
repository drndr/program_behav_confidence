"""Reusable program properties, starting with static size measurements.

Executable lines here means physical source lines with code tokens (including
definition/control headers and continuation lines), excluding blank/comment
lines and standalone string-literal statements such as docstrings. This is a
source-size convention, not a count of bytecode instructions or executed lines.
"""

import ast
import io
import re
import tokenize


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


def executable_lines(code, dataset):
    """Return count and diagnostics; incomplete tokenization produces no count."""
    labels = helper_label_lines(code) if dataset == "re2" else set()
    lines = code.splitlines(keepends=True)
    source = "".join("\n" if i in labels else line for i, line in enumerate(lines, 1))
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


def measure_size(example, encoding):
    """Tokenize the exact CSV code/input fields; no prompt wrapper or target."""
    return {"code_token_length": len(encoding.encode_ordinary(example["code"])),
            "input_token_length": len(encoding.encode_ordinary(example["input"])),
            **executable_lines(example["code"], example["dataset"])}


def measure_properties(example, encoding):
    """Collect property groups here as additional measurements are introduced."""
    return measure_size(example, encoding)
