"""Run one program under a line tracer and report its trace length (stdlib only, no project imports).

Executed as a child process by scripts/measure_execution.py, one program per process:

    python -s -P src/execution.py < payload.json

The payload holds the program's code, optional untraced setup statements, the traced call
expression (or none: the whole code is traced as a script), and optional standard input.
Trace length counts line events in the program's own code (each execution of a line, so a
loop body counts once per iteration); the harness and library code are not counted.
"""

import contextlib
import io
import json
import sys

PROGRAM = "<program>"


def trace(code, call=None, setup="", stdin=""):
    """Execute the code and return its trace length, return value and printed output.

    The return value comes as repr and as JSON with other objects given by their repr,
    the form RE2 records outputs in (None where it does not serialize).
    """
    namespace = {"__name__": "__main__" if call is None else "__program__"}
    lines = 0

    def tracer(frame, event, arg):
        nonlocal lines
        if frame.f_code.co_filename != PROGRAM:
            return None
        if event == "line":
            lines += 1
        return tracer

    printed, original_stdin = io.StringIO(), sys.stdin
    program = compile(code, PROGRAM, "exec")
    with contextlib.redirect_stdout(printed):
        sys.stdin = io.StringIO(stdin)
        try:
            result = _run(program, namespace, tracer, call, setup)
        finally:
            sys.stdin = original_stdin
    try:
        result_json = json.dumps(result, default=repr)
    except (ValueError, TypeError, RecursionError):
        result_json = None
    return {"trace_length": lines, "result": repr(result), "result_json": result_json, "printed": printed.getvalue()}


def _run(program, namespace, tracer, call, setup):
    """Trace the whole script, or run it and the setup untraced and trace only the call."""
    if call is None:
        sys.settrace(tracer)
        try:
            exec(program, namespace)
        except SystemExit:  # scripts may end with exit()
            pass
        finally:
            sys.settrace(None)
        return None
    exec(program, namespace)
    exec(compile(setup, "<setup>", "exec"), namespace)
    expression = compile(call, "<call>", "eval")
    sys.settrace(tracer)
    try:
        return eval(expression, namespace)
    finally:
        sys.settrace(None)


def limit_memory(megabytes=2048):
    """Cap the address space where the platform allows it (not on Windows)."""
    try:
        import resource
    except ImportError:
        return
    limit = megabytes * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


if __name__ == "__main__":
    payload = json.loads(sys.stdin.read())
    limit_memory()
    sys.setrecursionlimit(10_000)
    outcome = trace(payload["code"], payload.get("call"), payload.get("setup", ""), payload.get("stdin", ""))
    sys.stdout.write(json.dumps(outcome) + "\n")
