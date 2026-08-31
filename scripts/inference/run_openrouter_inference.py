"""Run resumable OpenRouter inference from a JSON configuration file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import time
from datetime import datetime, timezone
from http.client import IncompleteRead
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_OUTPUT_PATTERN = re.compile(r"Final output\s*:", re.IGNORECASE)
RETRYABLE_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
RESERVED_GENERATION_KEYS = {"messages", "model", "provider", "reasoning", "stream", "usage"}
SUPPORTED_PROMPT_MODES = {"auto", "direct", "cot"}
SUPPORTED_REASONING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}


class OpenRouterRequestError(RuntimeError):
    """An OpenRouter request failure with retry information."""

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="OpenRouter inference configuration JSON.")
    parser.add_argument("--prompt-path", type=Path, help="Override the configured prompt path.")
    parser.add_argument("--output-path", type=Path, help="Override the prediction output path.")
    parser.add_argument("--metadata-path", type=Path, help="Override the metadata output path.")
    parser.add_argument("--limit", type=int, help="Override the configured row limit.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(serialized)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed configuration JSON at {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("The inference configuration must be a JSON object")

    required = (
        "model_name",
        "model_revision",
        "api_url",
        "api_key_env",
        "expected_provider_name",
        "prompt_path",
        "output_path",
        "reasoning",
        "provider",
        "generation",
    )
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"Missing configuration fields: {missing}")

    if config.get("prompt_mode", "auto") not in SUPPORTED_PROMPT_MODES:
        raise ValueError("prompt_mode must be auto, direct, or cot")
    if config.get("limit") is not None and int(config["limit"]) <= 0:
        raise ValueError("limit must be null or a positive integer")
    if int(config.get("retries", 1)) < 0:
        raise ValueError("retries cannot be negative")
    if float(config.get("retry_backoff_seconds", 2.0)) < 0:
        raise ValueError("retry_backoff_seconds cannot be negative")
    if float(config.get("request_delay_seconds", 0.0)) < 0:
        raise ValueError("request_delay_seconds cannot be negative")
    if float(config.get("timeout_seconds", 300.0)) <= 0:
        raise ValueError("timeout_seconds must be positive")

    parsed_api_url = urlparse(str(config["api_url"]))
    if (
        parsed_api_url.scheme != "https"
        or parsed_api_url.hostname != "openrouter.ai"
        or parsed_api_url.path != "/api/v1/chat/completions"
    ):
        raise ValueError("api_url must be https://openrouter.ai/api/v1/chat/completions")

    api_key_env = config["api_key_env"]
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise ValueError("api_key_env must be a non-empty environment-variable name")
    if not api_key_env.startswith("OPENROUTER_"):
        raise ValueError("api_key_env must name an OPENROUTER_* environment variable")
    if not isinstance(config["expected_provider_name"], str) or not config[
        "expected_provider_name"
    ].strip():
        raise ValueError("expected_provider_name must be a non-empty string")

    reasoning = config["reasoning"]
    if not isinstance(reasoning, dict):
        raise ValueError("reasoning must be a JSON object")
    effort = reasoning.get("effort")
    if effort is None and reasoning.get("max_tokens") is None:
        raise ValueError("reasoning must explicitly set effort or max_tokens")
    if effort is not None and effort not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(f"Unsupported reasoning effort: {effort}")
    if reasoning.get("max_tokens") is not None and int(reasoning["max_tokens"]) <= 0:
        raise ValueError("reasoning.max_tokens must be positive")

    provider = config["provider"]
    if not isinstance(provider, dict):
        raise ValueError("provider must be a JSON object")
    provider_only = provider.get("only")
    if (
        not isinstance(provider_only, list)
        or not provider_only
        or not all(isinstance(value, str) and value for value in provider_only)
    ):
        raise ValueError("provider.only must be a non-empty list of provider slugs")
    if provider.get("allow_fallbacks") is not False:
        raise ValueError("provider.allow_fallbacks must be false for controlled experiments")
    if provider.get("require_parameters") is not True:
        raise ValueError("provider.require_parameters must be true for controlled experiments")

    generation = config["generation"]
    if not isinstance(generation, dict):
        raise ValueError("generation must be a JSON object")
    if int(generation.get("max_tokens", 0)) <= 0:
        raise ValueError("generation.max_tokens must be a positive integer")
    reserved = sorted(RESERVED_GENERATION_KEYS.intersection(generation))
    if reserved:
        raise ValueError(f"generation contains reserved request fields: {reserved}")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = dict(config)
    if args.prompt_path is not None:
        resolved["prompt_path"] = str(args.prompt_path)
    if args.output_path is not None:
        resolved["output_path"] = str(args.output_path)
    if args.metadata_path is not None:
        resolved["metadata_path"] = str(args.metadata_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        resolved["limit"] = args.limit
    return resolved


def infer_prompt_mode(record: dict[str, Any], requested_mode: str) -> str:
    stored_mode = str(record.get("prompt_mode", "")).lower()
    if requested_mode != "auto":
        if stored_mode in {"direct", "cot"} and stored_mode != requested_mode:
            raise RuntimeError(
                f"Prompt mode mismatch for {record.get('id')}: "
                f"record={stored_mode}, configured={requested_mode}"
            )
        return requested_mode
    if stored_mode in {"direct", "cot"}:
        return stored_mode
    prompt_version = str(record.get("prompt_version", "")).lower()
    return "cot" if "_cot_" in prompt_version else "direct"


def load_prompts(
    path: Path, requested_mode: str, limit: int | None
) -> tuple[list[dict[str, Any]], str]:
    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
    if limit is not None:
        records = records[: int(limit)]
    if not records:
        raise RuntimeError(f"No prompts found in {path}")

    ids = [str(record.get("id", "")) for record in records]
    if any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("Prompt IDs must be present and unique")

    modes = set()
    for record in records:
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError(f"Prompt messages are missing for {record['id']}")
        for message in messages:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
            ):
                raise RuntimeError(f"Invalid message for {record['id']}")
        if len(messages) == 1 and record.get("prompt_sha256"):
            actual_hash = sha256_text(messages[0]["content"])
            if actual_hash != record["prompt_sha256"]:
                raise RuntimeError(f"Prompt hash mismatch for {record['id']}")
        modes.add(infer_prompt_mode(record, requested_mode))
    if len(modes) != 1:
        raise RuntimeError(f"One run must contain one prompt mode, found {sorted(modes)}")
    return records, modes.pop()


def extract_prediction(response: str, mode: str) -> str:
    if mode != "cot":
        return response.strip()
    matches = list(FINAL_OUTPUT_PATTERN.finditer(response))
    if not matches:
        return response.strip()
    return response[matches[-1].end() :].strip()


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def reasoning_token_count(usage: dict[str, Any]) -> int | None:
    for details_key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(details_key)
        if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
            return int(details["reasoning_tokens"])
    if usage.get("reasoning_tokens") is not None:
        return int(usage["reasoning_tokens"])
    return None


def build_request_body(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    body = {
        "model": config["model_name"],
        "messages": record["messages"],
        "reasoning": config["reasoning"],
        "provider": config["provider"],
        "stream": False,
        "usage": {"include": True},
    }
    body.update(config["generation"])
    return body


def decode_error_body(raw_body: bytes) -> str:
    text = raw_body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text[:1000]
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        message = error.get("message")
        return f"{code}: {message}" if code else str(message)
    return text[:1000]


def post_json(
    api_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = Request(
        api_url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read()
    except HTTPError as exc:
        detail = decode_error_body(exc.read())
        raise OpenRouterRequestError(
            f"HTTP {exc.code}: {detail}",
            retryable=exc.code in RETRYABLE_HTTP_CODES,
            status_code=exc.code,
        ) from exc
    except (
        URLError,
        TimeoutError,
        socket.timeout,
        IncompleteRead,
        ConnectionError,
    ) as exc:
        raise OpenRouterRequestError(
            f"{type(exc).__name__}: {exc}",
            retryable=True,
        ) from exc

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise OpenRouterRequestError(
            f"Malformed JSON response: {exc}",
            retryable=True,
        ) from exc
    if not isinstance(payload, dict):
        raise OpenRouterRequestError("Response must be a JSON object", retryable=False)
    if payload.get("error"):
        error = payload["error"]
        raise OpenRouterRequestError(
            f"OpenRouter error: {error}",
            retryable=False,
        )
    return payload


def parse_completion(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise OpenRouterRequestError("Response has no completion choice", retryable=False)
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise OpenRouterRequestError("Completion choice has no message", retryable=False)
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return {
        "response": content_to_text(message.get("content")),
        "reasoning": message.get("reasoning"),
        "reasoning_details": message.get("reasoning_details"),
        "finish_reason": choice.get("finish_reason") or "unknown",
        "native_finish_reason": choice.get("native_finish_reason"),
        "response_id": payload.get("id"),
        "response_model": payload.get("model"),
        "provider": payload.get("provider"),
        "created": payload.get("created"),
        "usage": usage,
        "openrouter_metadata": payload.get("openrouter_metadata"),
    }


def validate_resolved_route(completion: dict[str, Any], config: dict[str, Any]) -> None:
    allowed_models = {config["model_name"], config["model_revision"]}
    response_model = completion.get("response_model")
    if response_model not in allowed_models:
        raise OpenRouterRequestError(
            f"Resolved model {response_model!r} is not one of {sorted(allowed_models)}",
            retryable=False,
        )

    response_provider = completion.get("provider")
    expected_provider = config["expected_provider_name"]
    if response_provider != expected_provider:
        raise OpenRouterRequestError(
            f"Resolved provider {response_provider!r} is not {expected_provider!r}",
            retryable=False,
        )

    metadata = completion.get("openrouter_metadata")
    if not isinstance(metadata, dict):
        raise OpenRouterRequestError("OpenRouter routing metadata is missing", retryable=False)
    endpoints = metadata.get("endpoints")
    available = endpoints.get("available") if isinstance(endpoints, dict) else None
    selected = (
        [endpoint for endpoint in available if isinstance(endpoint, dict) and endpoint.get("selected")]
        if isinstance(available, list)
        else []
    )
    if len(selected) != 1:
        raise OpenRouterRequestError(
            f"Expected one selected provider endpoint, found {len(selected)}",
            retryable=False,
        )
    selected_provider = selected[0].get("provider")
    selected_model = selected[0].get("model")
    if selected_provider != expected_provider or selected_model not in allowed_models:
        raise OpenRouterRequestError(
            "Router metadata does not match the configured model/provider: "
            f"provider={selected_provider!r}, model={selected_model!r}",
            retryable=False,
        )


def read_existing(
    path: Path, config_hash: str, prompt_ids: set[str]
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed checkpoint at {path}:{line_number}: {exc}") from exc
            sample_id = str(record.get("id", ""))
            if not sample_id or sample_id in records:
                raise RuntimeError(f"Missing or duplicate checkpoint ID at {path}:{line_number}")
            if record.get("config_hash") != config_hash:
                raise RuntimeError(f"Checkpoint configuration mismatch at {path}:{line_number}")
            records[sample_id] = record
    unknown = sorted(set(records) - prompt_ids)
    if unknown:
        raise RuntimeError(f"Checkpoint contains unknown prompt IDs: {unknown[:5]}")
    return records


def build_inference_config(
    config: dict[str, Any], mode: str, prompt_count: int, prompt_sha256: str
) -> dict[str, Any]:
    inference_config = {
        key: value
        for key, value in config.items()
        if key not in {"prompt_path", "output_path", "metadata_path"}
    }
    inference_config.update(
        prompt_mode=mode,
        prompt_count=prompt_count,
        prompt_file_sha256=prompt_sha256,
    )
    return inference_config


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_prompts(
    records: list[dict[str, Any]],
    mode: str,
    output_path: Path,
    config: dict[str, Any],
    config_hash: str,
    api_key: str,
) -> dict[str, dict[str, Any]]:
    prompt_ids = {str(record["id"]) for record in records}
    completed = read_existing(output_path, config_hash, prompt_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Resuming with {len(completed)}/{len(records)} completed prompts", flush=True)

    retries = int(config.get("retries", 1))
    backoff = float(config.get("retry_backoff_seconds", 2.0))
    request_delay = float(config.get("request_delay_seconds", 0.0))
    timeout = float(config.get("timeout_seconds", 300.0))

    with output_path.open("a", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            sample_id = str(record["id"])
            if sample_id in completed:
                continue

            completion: dict[str, Any] | None = None
            attempts = 0
            started = time.perf_counter()
            for attempt in range(retries + 1):
                attempts = attempt + 1
                try:
                    payload = post_json(
                        config["api_url"],
                        api_key,
                        build_request_body(record, config),
                        timeout,
                    )
                    completion = parse_completion(payload)
                    validate_resolved_route(completion, config)
                    break
                except OpenRouterRequestError as exc:
                    if attempt < retries and exc.retryable:
                        wait_seconds = backoff * (2**attempt)
                        print(
                            f"Retrying {sample_id} in {wait_seconds:.1f}s after {exc}",
                            flush=True,
                        )
                        time.sleep(wait_seconds)
                    else:
                        raise RuntimeError(
                            f"OpenRouter request failed for {sample_id} after "
                            f"{attempts} attempt(s): {exc}"
                        ) from exc
            latency = time.perf_counter() - started

            if completion is None:
                raise RuntimeError(f"OpenRouter returned no completion for {sample_id}")
            response = completion["response"]
            usage = completion["usage"]
            output_record = {
                "id": sample_id,
                "dataset": record.get("dataset", ""),
                "model": config["model_name"],
                "model_revision": config["model_revision"],
                "response_model": completion["response_model"],
                "provider": completion["provider"],
                "response_id": completion["response_id"],
                "response_created": completion["created"],
                "prompt_version": record.get("prompt_version", ""),
                "prompt_mode": mode,
                "prompt_sha256": record.get("prompt_sha256"),
                "response": response,
                "prediction": extract_prediction(response, mode),
                "reasoning": completion["reasoning"],
                "reasoning_details": completion["reasoning_details"],
                "target_output": record.get("target_output"),
                "finish_reason": completion["finish_reason"],
                "native_finish_reason": completion["native_finish_reason"],
                "prompt_token_count": usage.get("prompt_tokens") or usage.get("input_tokens"),
                "generated_token_count": usage.get("completion_tokens") or usage.get("output_tokens"),
                "reasoning_token_count": reasoning_token_count(usage),
                "total_token_count": usage.get("total_tokens"),
                "usage": usage,
                "openrouter_metadata": completion["openrouter_metadata"],
                "latency_seconds": round(latency, 6),
                "attempts": attempts,
                "config_hash": config_hash,
                "error": None,
            }
            output_file.write(json.dumps(output_record, ensure_ascii=False, sort_keys=True) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
            completed[sample_id] = output_record
            print(
                f"{len(completed)}/{len(records)} | {sample_id} | "
                f"tokens={output_record['generated_token_count']} | "
                f"reasoning={output_record['reasoning_token_count']} | {latency:.1f}s",
                flush=True,
            )
            if request_delay > 0:
                time.sleep(request_delay)
    return completed


def numeric_sum(records: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    total = 0.0
    for record in records:
        value: Any = record
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            total += float(value)
        elif isinstance(value, str):
            try:
                total += float(value)
            except ValueError:
                pass
    return total


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = apply_overrides(load_config(config_path), args)
    prompt_path = resolve_path(config["prompt_path"])
    output_path = resolve_path(config["output_path"])
    metadata_path = resolve_path(
        config.get("metadata_path", output_path.with_name(output_path.stem + "_metadata.json"))
    )
    records, mode = load_prompts(
        prompt_path, config.get("prompt_mode", "auto"), config.get("limit")
    )

    api_key_env = config["api_key_env"]
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {api_key_env} before running OpenRouter inference")

    inference_config = build_inference_config(
        config,
        mode,
        len(records),
        sha256_path(prompt_path),
    )
    config_hash = stable_hash(inference_config)
    read_existing(output_path, config_hash, {str(record["id"]) for record in records})
    metadata = {
        **inference_config,
        "config_file": str(config_path),
        "config_file_sha256": sha256_path(config_path),
        "config_hash": config_hash,
        "prompt_path": str(prompt_path),
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "started_at_utc": utc_now(),
        "status": "running",
        "runner_file_sha256": sha256_path(Path(__file__)),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    write_metadata(metadata_path, metadata)

    try:
        completed = run_prompts(records, mode, output_path, config, config_hash, api_key)
    except Exception as exc:
        completed = read_existing(
            output_path,
            config_hash,
            {str(record["id"]) for record in records},
        )
        metadata.update(
            status="failed",
            failed_at_utc=utc_now(),
            completed_rows=len(completed),
            failure=f"{type(exc).__name__}: {exc}",
        )
        write_metadata(metadata_path, metadata)
        raise
    completed_records = list(completed.values())
    metadata.update(
        status="complete",
        completed_at_utc=utc_now(),
        completed_rows=len(completed_records),
        error_rows=sum(bool(record.get("error")) for record in completed_records),
        truncated_rows=sum(
            record.get("finish_reason") == "length" for record in completed_records
        ),
        resolved_models=sorted(
            {
                str(record["response_model"])
                for record in completed_records
                if record.get("response_model")
            }
        ),
        resolved_providers=sorted(
            {
                str(record["provider"])
                for record in completed_records
                if record.get("provider")
            }
        ),
        total_prompt_tokens=int(numeric_sum(completed_records, ("prompt_token_count",))),
        total_generated_tokens=int(numeric_sum(completed_records, ("generated_token_count",))),
        total_reasoning_tokens=int(numeric_sum(completed_records, ("reasoning_token_count",))),
        reported_cost_usd=round(numeric_sum(completed_records, ("usage", "cost")), 8),
    )
    write_metadata(metadata_path, metadata)
    print(f"Predictions: {output_path.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
