"""Write prediction prompts for CRUXEval and the filtered RE2 set as JSONL.

    python scripts/generate_prompts.py            # both datasets
    python scripts/generate_prompts.py re2        # one dataset

Each row has id, dataset, entrypoint, messages and prompt_sha256. A manifest per
dataset records the source CSV hash and the prompt template hash, so results
can be tied to the exact wording that produced them.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import DATASET_SOURCES, load_examples, sha256_file
from prompt_template import prediction_prompt, prediction_template_sha256

SOURCES = DATASET_SOURCES


def generate(name, data_dir, out_dir, limit=None):
    source = data_dir / SOURCES[name]
    examples = load_examples(source, name)[:limit]
    rows = []
    for example in examples:
        prompt = prediction_prompt(example)
        rows.append({"id": example["id"], "dataset": name, "entrypoint": example["entrypoint"],
                     "messages": [{"role": "user", "content": prompt}],
                     "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{name}_prompts.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"dataset": name, "prompt_kind": "prediction", "count": len(rows),
                "source_path": (source.relative_to(ROOT) if source.is_relative_to(ROOT) else source).as_posix(),
                "source_sha256": sha256_file(source), "template_sha256": prediction_template_sha256(),
                "generated_at": datetime.now(timezone.utc).isoformat()}
    (out_dir / f"{name}_prompts.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", metavar="DATASET",
                        help=f"which to generate: {', '.join(SOURCES)} (default: all)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="data directory")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/prompts", help="output directory")
    parser.add_argument("--limit", type=int, help="only the first N examples, for pilots")
    args = parser.parse_args()
    selected = args.datasets or list(SOURCES)
    unknown = [name for name in selected if name not in SOURCES]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}; choose from {', '.join(SOURCES)}")
    for name in dict.fromkeys(selected):
        print(json.dumps(generate(name, args.data_dir, args.out_dir, args.limit), indent=2))


if __name__ == "__main__":
    main()
