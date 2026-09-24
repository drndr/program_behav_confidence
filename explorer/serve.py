"""Browse evaluated runs on localhost. Usage: python explorer/serve.py [PORT]"""

from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datasets import DATASET_SOURCES, load_examples
from runs import load_run

PAGE = Path(__file__).resolve().parent / "index.html"
EVALUATIONS = ROOT / "data/evaluations"
PREDICTIONS = ROOT / "data/predictions"


@lru_cache(maxsize=None)
def examples():
    return {(ds, r["id"]): r for ds, path in DATASET_SOURCES.items()
            for r in load_examples(ROOT / "data" / path, ds)}


def runs():
    return sorted(p.name.removesuffix("_scores.jsonl") for p in EVALUATIONS.glob("*_scores.jsonl"))


def reasoning_details(run):
    """Reasoning text and token count per item from the raw log, which evaluations do not keep.

    Empty when the log is absent or the run disabled reasoning; the latter skips
    reading a log that can be hundreds of megabytes.
    """
    path = PREDICTIONS / f"{run}.jsonl"
    if not path.exists():
        return {}
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if (manifest["configuration"]["parameters"].get("reasoning") or {}).get("enabled") is False:
        return {}
    details = {}
    for key, row in load_run(path)[1].items():
        try:
            reasoning = row["raw_response"]["choices"][0]["message"].get("reasoning")
        except (TypeError, KeyError, IndexError, AttributeError):
            reasoning = None
        tokens = ((row.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
        details[key] = {"reasoning": reasoning, "reasoning_tokens": tokens}
    return details


def items(run):
    """One record per item: code, input, reference, response, reasoning, answer logprob,
    reasoning tokens, and correct and status per scoring mode."""
    details = reasoning_details(run)
    grouped = {}
    with (EVALUATIONS / f"{run}_scores.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["dataset"], row["id"], row["repetition"])
            if key not in grouped:
                example = examples()[(row["dataset"], row["id"])]
                grouped[key] = {"dataset": row["dataset"], "id": row["id"], "repetition": row["repetition"],
                                "code": example["code"], "input": example["input"],
                                "reference": row["reference"], "response": row["prediction"],
                                "logprob": row["prediction_mean_logprob"],
                                "reasoning": None, "reasoning_tokens": None, **details.get(key, {}),
                                "correct": {}, "status": {}}
            grouped[key]["correct"][row["mode"]] = row["correct"]
            grouped[key]["status"][row["mode"]] = row["status"]
    return list(grouped.values())


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/runs":
            self.send(json.dumps(runs()).encode(), "application/json")
        elif self.path.startswith("/api/runs/") and self.path.removeprefix("/api/runs/") in runs():
            self.send(json.dumps(items(self.path.removeprefix("/api/runs/"))).encode(), "application/json")
        else:
            self.send_error(404)

    def send(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Explorer on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    server.serve_forever()
