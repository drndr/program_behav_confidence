"""Run saved prompts through OpenRouter; save each response and resume safely.

Set OPENROUTER_API_KEY in the environment or the root .env file, then run:
python scripts/run_predictions.py --model ORGANIZATION/MODEL --run-name pilot --limit 10
Use --dry-run first (no key or network needed). --limit applies per dataset.
Run the same command again to resume. Failed requests are retried only with
--retry-errors; incomplete completions are retained and not automatically repeated.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datasets import sha256_file
from model_client import OpenRouterClient
from scoring import parse_prediction, prediction_logprobs

NETWORK_FAILURE_LIMIT = 3


def validate_logprobs(result):
    """Keep response status intact; independently validate usable answer logprobs."""
    if result["status"] != "completed":
        return {"valid": False, "reason": "response_not_completed"}
    logprobs = result.get("logprobs")
    if not isinstance(logprobs, dict) or not logprobs.get("content"):
        return {"valid": False, "reason": "missing_or_unaligned_logprobs"}
    try:
        parse_prediction(result.get("text") or "")
    except ValueError:
        return {"valid": False, "reason": "invalid_answer_format"}
    try:
        metrics = prediction_logprobs(result.get("text") or "", result.get("logprobs"))
        if metrics is None:
            return {"valid": False, "reason": "missing_or_unaligned_logprobs"}
        if not math.isfinite(metrics["mean_logprob"]) or metrics["mean_logprob"] > 0:
            return {"valid": False, "reason": "invalid_logprob_value"}
    except (ValueError, TypeError, KeyError, OverflowError, AttributeError):
        return {"valid": False, "reason": "malformed_answer_or_logprobs"}
    return {"valid": True, **metrics}


def logprobs_block_run(result):
    """A completed answer without usable logprobs stops the run; failed or truncated requests do not."""
    if result["status"] != "completed":
        return False
    validation = validate_logprobs(result)
    return not validation["valid"] and validation.get("reason") != "invalid_answer_format"


def now():
    return datetime.now(timezone.utc).isoformat()


def load_prompts(paths, limit):
    rows, seen = [], set()
    for path in paths:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in records[:limit]:
            key = (row["dataset"], row["id"])
            if key in seen:
                raise ValueError(f"Duplicate prompt: {key}")
            seen.add(key)
            messages = row["messages"]
            # Current generated protocol has exactly one user message.
            if len(messages) != 1 or messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), str):
                raise ValueError(f"Invalid prompt messages: {key}")
            digest = hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest()
            if digest != row["prompt_sha256"]:
                raise ValueError(f"Prompt hash mismatch: {key}")
            rows.append(row)
    if not rows:
        raise ValueError("No prompts selected")
    return rows


def read_results(path):
    """Recover a torn final write; preserve its bytes in a separate file."""
    if not path.exists():
        return []
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    rows, offset = [], 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt result log at line {index + 1}")
            backup = path.with_name(path.name + ".partial")
            with backup.open("ab") as stream:
                stream.write(line + b"\n")
            with path.open("r+b") as stream:
                stream.truncate(offset)
            break
        rows.append(row)
        offset += len(line)
    if rows and offset == len(content) and not content.endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return rows


def run(args, client=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("run-name must be a simple filename")
    if args.repetitions < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Repetitions and limit must be positive")
    if args.max_tokens < 1 or args.timeout <= 0:
        raise ValueError("max-tokens and timeout must be positive")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if args.allow_unlisted_parameters and not args.backend:
        raise ValueError("--allow-unlisted-parameters needs --backend")
    paths = [args.prompt_dir / f"{name}_prompts.jsonl" for name in args.datasets]
    prompts = load_prompts(paths, args.limit)
    parameters = {"max_tokens": args.max_tokens, "provider": {"require_parameters": not args.allow_unlisted_parameters}}
    if args.temperature is not None:
        parameters["temperature"] = args.temperature
    if args.logprobs or args.require_logprobs:
        parameters["logprobs"] = True
    if args.backend:
        parameters["provider"].update(only=[args.backend], allow_fallbacks=False)
    if args.reasoning_effort is not None:
        parameters["reasoning"] = {"effort": args.reasoning_effort}
        if args.reasoning_effort == "none":
            parameters["reasoning"]["enabled"] = False
    config = {"provider": "openrouter", "model": args.model, "parameters": parameters,
              "timeout": args.timeout, "repetitions": args.repetitions, "limit_per_dataset": args.limit,
              "prompt_files": {p.name: sha256_file(p) for p in paths}}
    if args.require_logprobs:
        config["require_logprobs"] = True
    request_count = len(prompts) * args.repetitions
    if args.dry_run:
        print(json.dumps({"requests": request_count, "configuration": config}, indent=2))
        return {"planned": request_count}
    if client is None:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False, interpolate=False)
        client = OpenRouterClient(timeout=args.timeout)
    if client.provider != config["provider"]:
        raise ValueError("Client provider differs from run configuration")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = args.out_dir / f"{args.run_name}.jsonl"
    manifest = args.out_dir / f"{args.run_name}.manifest.json"
    lock = args.out_dir / f"{args.run_name}.lock"
    # Prevent two writers from issuing the same requests. A hard-killed process may
    # leave this lock; remove it only after checking that its process has stopped.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        if manifest.exists():
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            if previous["configuration"] != config:
                raise ValueError("Run configuration changed; choose a new run-name")
        else:
            if log.exists():
                raise ValueError("Existing result log has no manifest")
            manifest.write_text(json.dumps({"created_at": now(), "configuration": config,
                                "planned_requests": request_count}, indent=2) + "\n", encoding="utf-8")
        history = read_results(log)
        latest, attempts = {}, Counter()
        for row in history:
            key = (row["dataset"], row["id"], row["repetition"])
            latest[key] = row
            attempts[key] += 1
        counts, network_failures = Counter(), 0
        with log.open("a", encoding="utf-8", newline="\n") as stream:
            for prompt in prompts:
                for repetition in range(args.repetitions):
                    key = (prompt["dataset"], prompt["id"], repetition)
                    prior = latest.get(key)
                    if prior and (prior["status"] != "error" or not args.retry_errors):
                        if args.require_logprobs and logprobs_block_run(prior):
                            raise ValueError("Saved response fails required logprob validation; inspect it before resuming")
                        counts["skipped"] += 1
                        continue
                    result = client.complete(prompt["messages"], args.model, parameters)
                    if args.require_logprobs:
                        result["logprob_validation"] = validate_logprobs(result)
                    record = {"dataset": prompt["dataset"], "id": prompt["id"], "repetition": repetition,
                              "attempt": attempts[key] + 1, "prompt_sha256": prompt["prompt_sha256"],
                              "provider": client.provider, "requested_model": args.model,
                              "request_parameters": parameters, "saved_at": now(), **result}
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    counts[result["status"]] += 1
                    print(f"{prompt['dataset']} {prompt['id']} repeat={repetition}: {result['status']}", flush=True)
                    if args.require_logprobs and logprobs_block_run(result):
                        raise ValueError("Saved response fails required logprob validation: "
                                         + result["logprob_validation"]["reason"])
                    if result.get("http_status") in (401, 402, 403, 429):
                        print("Stopping on authentication, credit, permission or rate-limit error; see saved response.")
                        return dict(counts)
                    # No HTTP response at all (DNS, connection, timeout): stop rather than fail every prompt.
                    network_failures = network_failures + 1 if result["status"] == "error" and result.get("http_status") is None else 0
                    if network_failures >= NETWORK_FAILURE_LIMIT:
                        print(f"Stopping after {NETWORK_FAILURE_LIMIT} consecutive network failures; "
                              "check the connection and resume with --retry-errors.")
                        return dict(counts)
        print(json.dumps(dict(counts)))
        return dict(counts)
    finally:
        lock.unlink()


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Exact OpenRouter model ID, including organization prefix")
    p.add_argument("--run-name", required=True)
    p.add_argument("--datasets", nargs="+", choices=("cruxeval", "re2"), default=["cruxeval", "re2"])
    p.add_argument("--prompt-dir", type=Path, default=ROOT / "data/prompts")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/predictions")
    p.add_argument("--limit", type=int, help="First N examples per dataset")
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, help="Omit to use the model/provider default")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--logprobs", action="store_true")
    p.add_argument("--require-logprobs", action="store_true",
                   help="Require usable answer logprobs; retain and skip incomplete responses")
    p.add_argument("--backend", help="Restrict OpenRouter to one backend provider slug")
    p.add_argument("--allow-unlisted-parameters", action="store_true",
                   help="Send parameters OpenRouter does not list for the backend (it omits logprobs for some "
                        "models); needs --backend, and --require-logprobs checks each response")
    p.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high", "xhigh"))
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
