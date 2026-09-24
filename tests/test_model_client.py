import io
import json
from http.client import IncompleteRead
from pathlib import Path
import sys
import unittest
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_client import OpenRouterClient


class Reply(io.BytesIO):
    status = 200


class IncompleteReply(Reply):
    def read(self, *args, **kwargs):
        raise IncompleteRead(b"partial")


class ClientTests(unittest.TestCase):
    def test_request_and_raw_response_preserved(self):
        logprobs = {"content": [{"token": "1", "logprob": -0.5}]}
        raw = {"id": "generation-1", "model": "resolved/model", "provider": "backend",
               "choices": [{"message": {"content": "<prediction>1</prediction>", "reasoning": "trace"},
                            "finish_reason": "stop", "logprobs": logprobs}],
               "usage": {"total_tokens": 10, "cost": 0.001}}
        def transport(request, timeout):
            payload = json.loads(request.data)
            self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
            self.assertEqual(payload["model"], "requested/model")
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["messages"], [{"role": "user", "content": "question"}])
            return Reply(json.dumps(raw).encode())
        client = OpenRouterClient("test-secret", transport=transport)
        result = client.complete([{"role": "user", "content": "question"}], "requested/model", {})
        # Log-probabilities are stored once, in their own field; the rest of the response is kept verbatim.
        self.assertEqual(result["logprobs"], logprobs)
        self.assertNotIn("logprobs", result["raw_response"]["choices"][0])
        expected = json.loads(json.dumps(raw))
        del expected["choices"][0]["logprobs"]
        self.assertEqual(result["raw_response"], expected)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["backend_provider"], "backend")
        self.assertNotIn("test-secret", json.dumps(result))

    def test_truncated_response_is_retained(self):
        raw = {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
        result = OpenRouterClient("test-secret", transport=lambda *a, **k: Reply(json.dumps(raw).encode())).complete([], "m", {})
        self.assertEqual((result["status"], result["text"]), ("incomplete", "partial"))

    def test_http_and_in_band_errors(self):
        def http_error(*args, **kwargs):
            raise HTTPError("url", 429, "limited", {}, io.BytesIO(b'{"error":{"message":"limited"}}'))
        result = OpenRouterClient("test-secret", transport=http_error).complete([], "m", {})
        self.assertEqual((result["status"], result["http_status"]), ("error", 429))
        result = OpenRouterClient("test-secret", transport=lambda *a, **k: Reply(b'{"error":{"code":500}}')).complete([], "m", {})
        self.assertEqual(result["status"], "error")

    def test_transport_failure_does_not_expose_key(self):
        def fail(*a, **k): raise URLError("test-secret")
        result = OpenRouterClient("test-secret", transport=fail).complete([], "m", {})
        self.assertEqual(result["status"], "error")
        self.assertNotIn("test-secret", json.dumps(result))

    def test_incomplete_http_response_is_recorded_as_error(self):
        result = OpenRouterClient("test-secret", transport=lambda *a, **k: IncompleteReply()).complete([], "m", {})
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["raw_response"])


if __name__ == "__main__":
    unittest.main()
