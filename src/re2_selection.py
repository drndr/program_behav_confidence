"""Apply the reviewed RE2 exclusions and seven upstream-backed JSON repairs."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import zipfile

from datasets import load_re2_export, sha256_file
from re2_inspection import input_text
from re2_filter import filter_re2


def prepare_re2(data_dir):
    """Return (selected rows, inspection report); do not modify raw files.

    This is a screened selection, not independent validation of every reference.
    Repairs restore upstream JSON values, including serialized object strings.
    Avatar stdin normalization is outside this function's scope.
    """
    data_dir = Path(data_dir)
    provenance = json.loads((data_dir / "provenance.json").read_text(encoding="utf-8"))
    source = next(p for p in provenance if "/re2/" in p["path"])
    raw_path = data_dir.parent / source["path"]
    archive_path = data_dir.parent / source["source_path"]
    invalid = json.loads((data_dir / "re2_invalid_input_review.json").read_text(encoding="utf-8"))
    valid = json.loads((data_dir / "re2_valid_json_review.json").read_text(encoding="utf-8"))
    digest = sha256_file(raw_path)
    if digest != invalid["export_sha256"] or digest != valid["source_sha256"] or digest != source["sha256"]:
        raise ValueError("Review/source hash mismatch; review this dataset version first")
    if sha256_file(archive_path) != source["source_sha256"]:
        raise ValueError("Upstream archive hash mismatch")
    rows = load_re2_export(raw_path)
    repairs = {c["id"]: c for c in invalid["cases"] if c["recommendation"] == "repair_candidate"}
    retained, excluded = filter_re2(rows)
    selected, repaired = [], []
    with zipfile.ZipFile(archive_path) as archive:
        for original in retained:
            case_id = original["id"]
            row = deepcopy(original)
            if case_id in repairs:
                review = repairs[case_id]
                upstream = json.loads(archive.read(review["upstream_record_path"]).decode("utf-8").splitlines()[review["upstream_io_index"]])
                values = upstream["values"]["inputs"]
                replacement = json.dumps(values, indent=4, ensure_ascii=False, allow_nan=False)
                prompt = row["messages"][0]["content"]
                match = list(re.finditer(r"\[INPUT\](.*?)\[/INPUT\]", prompt, re.S))[-1]
                start, end = match.span(1)
                row["messages"][0]["content"] = prompt[:start] + "\n```json\n" + replacement + "\n```\n" + prompt[end:]
                row["original_prompt_sha256"] = original["prompt_sha256"]
                row["prompt_sha256"] = hashlib.sha256(row["messages"][0]["content"].encode("utf-8")).hexdigest()
                row["input_repair"] = "restore_upstream_inputs_as_valid_json"
                repaired.append({"id": case_id, "before": input_text(original), "after": input_text(row),
                                 "evidence": review["evidence"], "upstream_path": review["upstream_record_path"],
                                 "upstream_io_index": review["upstream_io_index"]})
                if json.loads(input_text(row)) != values:
                    raise ValueError(f"Repair differs from upstream: {case_id}")
            json.loads(input_text(row))  # Fail rather than retain unexpected malformed inputs.
            selected.append(row)
    counts = dict(Counter(r["reason"] for r in excluded))
    if (len(rows), len(selected), len(repaired), counts) != (500, 446, 7, {
        "cruxeval_overlap": 44, "problematic_sample": 10
    }):
        raise ValueError("Selection differs from the reviewed 500-record release")
    return selected, {"source_sha256": digest, "input_count": len(rows), "retained_count": len(selected),
                      "excluded_count": len(excluded), "repaired_count": len(repaired),
                      "exclusion_counts": counts, "excluded": excluded, "repairs": repaired,
                      "limitations": "Screened, not fully validated. Object representations and targets unchanged. Avatar stdin encoding unchanged."}


def write_re2_selection(rows, report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "re2_screened.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (output_dir / "selection_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
