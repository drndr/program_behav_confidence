"""Read saved model runs."""

import json
from pathlib import Path


def load_run(path):
    """Return the run manifest and its latest row per (dataset, id, repetition).

    The runner appends on retries, so a log can hold several attempts for one item.
    """
    path = Path(path)
    manifest = json.loads(path.with_name(path.stem + ".manifest.json").read_text(encoding="utf-8"))
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(row["dataset"], row["id"], row["repetition"])] = row
    return manifest, latest
