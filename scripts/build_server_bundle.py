"""Zip the Qwen server notebook with the prompts and the code it reuses; no model calls.

    python scripts/build_server_bundle.py

Writes _local/server_bundles/qwen38_27b_server.zip (see server/README.md). Files keep their
repository paths, so on the server the notebook imports the runner's prompt checks and the
verification prompt script, and writes results where the repository keeps them.
bundle_manifest.json records the commit, bundled files that differ from it, and the hash of
every file the notebook reads; the notebook checks those hashes before running.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datasets import sha256_file

NOTEBOOK = "server/qwen38_27b_transformers.ipynb"
README = "server/README.md"
# What the notebook reads: the prompts, the datasets verification prompts are built from, and
# the modules it imports (with their own imports).
INPUTS = ["data/prompts/cruxeval_prompts.jsonl", "data/prompts/cruxeval_prompts.manifest.json",
          "data/prompts/re2_prompts.jsonl", "data/prompts/re2_prompts.manifest.json",
          "data/prompts/re2_additions/re2_prompts.jsonl",
          "data/raw/cruxeval/cruxeval.csv", "data/raw/re2/re2_filtered.csv",
          "scripts/run_predictions.py", "scripts/generate_verification_prompts.py",
          "src/datasets.py", "src/model_client.py", "src/prompt_template.py", "src/runs.py", "src/scoring.py"]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def build(out):
    out = Path(out)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git("rev-parse", "HEAD"),
                "uncommitted": git("status", "--porcelain", "--", NOTEBOOK, README, *INPUTS).splitlines(),
                "notebook_sha256": sha256_file(ROOT / NOTEBOOK),
                "files": {path: sha256_file(ROOT / path) for path in INPUTS}}
    out.parent.mkdir(parents=True, exist_ok=True)
    folder = out.stem
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / NOTEBOOK, f"{folder}/{Path(NOTEBOOK).name}")
        archive.write(ROOT / README, f"{folder}/README.md")
        for path in INPUTS:
            archive.write(ROOT / path, f"{folder}/{path}")
        archive.writestr(f"{folder}/bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "_local/server_bundles/qwen38_27b_server.zip")
    args = parser.parse_args()
    manifest = build(args.out)
    print(f"Wrote {args.out} from commit {manifest['git_commit'][:7]}")
    if manifest["uncommitted"]:
        print("Bundled files not committed:", *manifest["uncommitted"], sep="\n  ")
