"""Separate provider clients share complete(); currently implements OpenRouter only."""

import json
import os
import time
from http.client import IncompleteRead
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelClient(Protocol):
    provider: str

    def complete(self, messages, model, parameters):
        """Return status, raw response, usage, latency and provider metadata."""
        ...


class OpenRouterClient:
    provider = "openrouter"
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key=None, timeout=120, transport=None):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("Set OPENROUTER_API_KEY in the environment")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self._transport = transport or urlopen

    def complete(self, messages, model, parameters):
        if set(parameters) & {"model", "messages", "stream", "n"}:
            raise ValueError("Parameters cannot override model, messages, stream or n")
        payload = {**parameters, "model": model, "messages": messages, "stream": False}
        request = Request(self.endpoint, data=json.dumps(payload, allow_nan=False).encode("utf-8"),
                          headers={"Authorization": f"Bearer {self._api_key}",
                                   "Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        try:
            with self._transport(request, timeout=self.timeout) as response:
                status_code = response.status
                raw_text = response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            status_code = error.code
            raw_text = error.read().decode("utf-8", errors="replace")
        except (IncompleteRead, URLError, TimeoutError, OSError) as error:
            return {"status": "error", "error": str(error).replace(self._api_key, "[REDACTED]"),
                    "latency_seconds": time.perf_counter() - started,
                    "http_status": None, "raw_response": None}
        latency = time.perf_counter() - started
        raw_text = raw_text.replace(self._api_key, "[REDACTED]")
        try:
            raw = json.loads(raw_text)
        except ValueError:
            return {"status": "error", "error": "Non-JSON API response", "http_status": status_code,
                    "latency_seconds": latency, "raw_response": raw_text}
        result = {"status": "error", "http_status": status_code, "latency_seconds": latency,
                  "raw_response": raw}
        if status_code >= 400 or not isinstance(raw, dict) or raw.get("error"):
            return dict(result, error="API error; see raw_response")
        choices = raw.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            return dict(result, error="Expected one completion choice")
        choice = choices[0]
        message = choice.get("message")
        if choice.get("error") or not isinstance(message, dict):
            return dict(result, error="Invalid completion message; see raw_response")
        content = message.get("content")
        finish = choice.get("finish_reason")
        return dict(result,
                    status="completed" if finish == "stop" and isinstance(content, str) and content.strip() else "incomplete",
                    text=content, finish_reason=finish, usage=raw.get("usage"),
                    logprobs=choice.get("logprobs"), response_id=raw.get("id"),
                    returned_model=raw.get("model"), backend_provider=raw.get("provider"))
