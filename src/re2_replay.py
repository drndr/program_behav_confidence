"""Conservative, capability-restricted replay of self-contained RE2 snippets.

Unsupported constructs are unresolved, never benchmark defects. This is not a
general Python sandbox: only AST-screened snippets are admitted, in a separate
process with a parent timeout. No filesystem/network APIs are made available.
"""

import ast
import builtins
import contextlib
import importlib
import inspect
import io
import json
import sys


MODULES = {"math", "re", "typing", "collections", "decimal", "itertools", "functools", "string"}
BUILTINS = {"abs", "all", "any", "bin", "bool", "bytearray", "bytes", "chr", "complex", "dict",
            "divmod", "enumerate", "filter", "float", "hex", "int", "isinstance", "issubclass",
            "iter", "len", "list", "map", "max", "min", "next", "oct", "ord", "range",
            "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
            "zip", "staticmethod", "classmethod", "property", "object",
            "Exception", "ValueError", "TypeError", "IndexError", "KeyError", "StopIteration"}
ATTRS = set("append extend insert pop remove clear copy count index reverse sort add discard update "
            "get keys values items setdefault join split rsplit strip lstrip rstrip replace lower upper "
            "capitalize title startswith endswith isalpha isdigit isalnum isspace isupper islower "
            "encode decode format find rfind translate maketrans real imag conjugate "
            "ceil floor sqrt sin cos tan pi e inf log log2 log10 exp fabs factorial gcd isclose "
            "compile match fullmatch search findall finditer sub group groups start end span "
            "most_common elements appendleft popleft rotate "
            "ascii_lowercase ascii_uppercase ascii_letters digits punctuation whitespace "
            "prec quantize normalize to_integral_value".split())


def replay(payload):
    tree = ast.parse(payload["code"])
    defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            if getattr(node, "level", 0) or any(m not in MODULES for m in modules):
                return {"status": "unresolved", "reason": "unsupported_import", "detail": ast.unparse(node)}
            imported.update(a.asname or a.name.split('.')[0] for a in node.names)
        if isinstance(node, (ast.With, ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal)):
            return {"status": "unresolved", "reason": "unsupported_construct", "detail": type(node).__name__}
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return {"status": "unresolved", "reason": "unsupported_attribute", "detail": node.attr}
            if not (isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}) and node.attr not in ATTRS | defined:
                return {"status": "unresolved", "reason": "unsupported_attribute", "detail": node.attr}
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return {"status": "unresolved", "reason": "unsupported_name", "detail": node.id}
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in BUILTINS | defined | imported | {"input", "print"}:
                return {"status": "unresolved", "reason": "unsupported_call", "detail": node.func.id}
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                return {"status": "unresolved", "reason": "indirect_call"}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, (int, float)) or abs(node.right.value) > 32:
                return {"status": "unresolved", "reason": "unbounded_exponent"}
    safe = {name: getattr(builtins, name) for name in BUILTINS}

    def allowed_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in MODULES or any(x.startswith('_') for x in (fromlist or ())):
            raise ValueError("Import not permitted")
        return importlib.import_module(name)

    def bounded_range(*args):
        result = range(*args)
        if len(result) > 100000:
            raise ValueError("Range exceeds replay limit")
        return result

    output = io.StringIO()
    stdin = io.StringIO(payload.get("stdin", ""))

    def local_input(prompt=""):
        line = stdin.readline()
        if not line:
            raise EOFError("End of input")
        return line.rstrip("\n")

    def local_print(*args, sep=" ", end="\n"):
        text = sep.join(str(x) for x in args) + end
        if len(text) + output.tell() > 100000:
            raise ValueError("Output exceeds replay limit")
        output.write(text)

    safe.update(__import__=allowed_import, __build_class__=builtins.__build_class__, range=bounded_range,
                input=local_input, print=local_print)
    namespace = {"__builtins__": safe, "__name__": "re2_replay"}
    steps = 0

    def trace(frame, event, arg):
        nonlocal steps
        if event in {"line", "call"}:
            steps += 1
            if steps > 200000:
                raise RuntimeError("Replay step limit")
        return trace

    sys.settrace(trace)
    try:
        exec(compile(tree, "<re2>", "exec"), namespace)
        if payload["source"] == "Avatar":
            return {"status": "replayed", "stdout": output.getvalue(), "steps": steps}
        target = payload["entrypoint"]
        inputs = payload["inputs"]
        if target in namespace:
            function = namespace[target]
        else:
            classes = [v for v in namespace.values() if isinstance(v, type) and v.__module__ == "re2_replay" and target in v.__dict__]
            if len(classes) != 1:
                return {"status": "unresolved", "reason": "entrypoint_not_resolved"}
            cls = classes[0]
            descriptor = inspect.getattr_static(cls, target)
            if isinstance(descriptor, staticmethod):
                function = getattr(cls, target)
            elif isinstance(descriptor, classmethod):
                function = getattr(cls, target)
            else:
                obj = object.__new__(cls)
                for name, value in inputs.get("self", {}).items():
                    if name.startswith("__"):
                        return {"status": "unresolved", "reason": "private_state_encoding"}
                    setattr(obj, name, value)
                function = getattr(obj, target)
        arguments = dict(inputs.get("args", {}))
        extra = inputs.get("kwargs", {})
        if set(arguments) & set(extra):
            return {"status": "unresolved", "reason": "duplicate_arguments"}
        signature = inspect.signature(function)
        positional = []
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL and arguments.get(p.name)
                          for p in signature.parameters.values())
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                positional.extend(arguments.pop(parameter.name, []))
            elif parameter.kind == inspect.Parameter.POSITIONAL_ONLY or (has_varargs and parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
                if parameter.name in arguments:
                    positional.append(arguments.pop(parameter.name))
                elif parameter.default is not inspect.Parameter.empty:
                    positional.append(parameter.default)
                else:
                    return {"status": "unresolved", "reason": "missing_required_argument", "detail": parameter.name}
        signature.bind(*positional, **arguments, **extra)
        value = function(*positional, **arguments, **extra)
        # Upstream JSON represents tuples as lists. No string/numeric coercion.
        value = json.loads(json.dumps(value, allow_nan=False))
        return {"status": "replayed", "value": value, "steps": steps}
    except Exception as error:
        return {"status": "unresolved", "reason": "replay_exception", "detail": f"{type(error).__name__}: {error}"[:400]}
    finally:
        sys.settrace(None)


if __name__ == "__main__":
    try:
        print(json.dumps(replay(json.load(sys.stdin))))
    except Exception as error:
        print(json.dumps({"status": "unresolved", "reason": "replay_exception", "detail": f"{type(error).__name__}: {error}"[:400]}))
