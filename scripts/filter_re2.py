"""Filter RE2, reduce prompts to code, entrypoint and input, and export full and filtered task CSVs."""

import csv
from collections import Counter
from copy import deepcopy
import hashlib
import zipfile
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_re2_export, sha256_file


# The task question names the called function; stdin programs have no name.
QUESTION = re.compile(r"What will be the output of (?:`([^`]+)`,? given )?the following input:")


def input_text(row):
    """The final INPUT block belongs to the task in this release's templates."""
    blocks = re.findall(r"\[INPUT\](.*?)\[/INPUT\]", row["messages"][0]["content"], re.S)
    if not blocks:
        raise ValueError(f"No input block: {row['id']}")
    return re.sub(r"^```(?:json|python)?\s*|\s*```$", "", blocks[-1].strip()).strip()


def remove_prompts(rows):
    """Return id/code/entrypoint/input/output records, retaining task and helper code.

    entrypoint is the function or method the task calls, or "" for programs
    that read stdin. The name is taken from the prompt because ids do not
    always carry it (HumanEval), and it is checked against the extracted code.
    """
    cleaned = []
    for row in rows:
        prompt = row["messages"][0]["content"].rsplit("[/EXAMPLE]", 1)[-1]
        main, _, helpers = prompt.partition("Functions called during the execution:")
        blocks = re.findall(r"\[PYTHON\](.*?)\[/PYTHON\]", main, re.S)
        if not blocks:
            raise ValueError(f"Missing task code: {row['id']}")
        code = blocks[-1].strip()
        helper_blocks = re.findall(r"\[PYTHON\](.*?)\[/PYTHON\]", helpers, re.S)
        if helper_blocks:
            code += "\n\n" + "\n\n".join(b.strip() for b in helper_blocks)
        questions = QUESTION.findall(prompt)  # The last one belongs to the task, not a worked example.
        if not questions:
            raise ValueError(f"Missing task question: {row['id']}")
        entrypoint = questions[-1]
        task_input = input_text(row)
        if entrypoint:
            if not re.search(r"\bdef\s+" + re.escape(entrypoint) + r"\s*\(", code):
                raise ValueError(f"Entrypoint {entrypoint} is not defined in the task code: {row['id']}")
        else:
            try:
                stdin_only = set(json.loads(task_input)) == {"input"}
            except (ValueError, TypeError):
                stdin_only = False
            if not stdin_only:
                raise ValueError(f"Program without entrypoint must take stdin input only: {row['id']}")
        cleaned.append({"id": row["id"], "code": code, "entrypoint": entrypoint,
                        "input": task_input, "output": row["target_output"]})
    return cleaned

PROBLEMATIC_CASES = {
    "matplotlib@@matplotlib__api_deprecation.py@@deprecated_L97": "Unpredictable function address",
    "matplotlib@@matplotlib__api_deprecation.py@@delete_parameter_L314": "Unpredictable function address",
    "sympy@@sympy_multipledispatch_dispatcher.py@@register_L120": "Unpredictable function address",
    "matplotlib@@matplotlib___init__.py@@_preprocess_data_L1432": "Unpredictable function address",
    "matplotlib@@matplotlib_colors.py@@_repr_png__L2115": "Missing pixels in abbreviated input array",
    "matplotlib@@matplotlib_colors.py@@_repr_html__L2140": "Missing pixels in abbreviated input array",
    "scikit-learn@@sklearn_ensemble__bagging.py@@predict_log_proba_L1055": "Missing fitted estimator state",
    "ClassEval_49@JobMarketplace.withdraw_resume": "Empty resume list raises an exception; reference is null",
    "ClassEval_95@Warehouse.change_order_status": "Integer ID and string dictionary key disagree with reference",
    "ClassEval_95@Warehouse.create_order": "Integer ID and string dictionary key disagree with reference",
}


def filter_re2(rows):
    """Return retained rows and exclusions, using source tags and explicit IDs."""
    kept, excluded = [], []
    for row in rows:
        overlap = row["source_dataset"] == "cruxeval"
        problem = PROBLEMATIC_CASES.get(row["id"])
        if overlap or problem:
            excluded.append({"id": row["id"],
                             "reason": "cruxeval_overlap" if overlap else "problematic_sample",
                             "evidence": "CRUXEval source" if overlap else problem})
        else:
            kept.append(row)
    return kept, excluded


def write_re2_csv(rows, path):
    """Write already-cleaned id/code/entrypoint/input/output records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "code", "entrypoint", "input", "output"])
        writer.writeheader()
        writer.writerows(rows)


def prepare_re2(data_dir):
    """Return (selected rows, inspection report); do not modify raw files.

    This is a screened selection, not independent validation of every reference.
    Repairs restore upstream JSON values, including serialized object strings.
    Avatar stdin is restored from upstream strings or numeric values as text.
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
            review = None
            kind = None
            if case_id in repairs:
                review = repairs[case_id]
                upstream = json.loads(archive.read(review["upstream_record_path"]).decode("utf-8").splitlines()[review["upstream_io_index"]])
                values = upstream["values"]["inputs"]
                kind = "malformed_json"
            elif original["source_dataset"] == "Avatar":
                path = (f"RE2-Bench-{original['artifact_commit']}/dataset/re2-bench/"
                        f"input-output/Avatar/{original['upstream_case_name']}")
                upstream = json.loads(archive.read(path).decode("utf-8").splitlines()[original["upstream_io_index"]])["values"]["input"]
                before = json.loads(input_text(original))
                if before != {"input": json.dumps(upstream, indent=4)}:
                    raise ValueError(f"Unexpected Avatar encoding: {case_id}")
                if isinstance(upstream, str):
                    stdin = upstream  # Preserve whitespace and newlines exactly.
                    kind = "avatar_string_decode"
                elif type(upstream) in (int, float):
                    stdin = json.dumps(upstream, allow_nan=False)
                    kind = "avatar_numeric_text"
                else:
                    raise ValueError(f"Unsupported Avatar stdin type: {case_id}")
                values = {"input": stdin}
                review = {"evidence": "Restore upstream stdin as text; numeric inputs already have this value.",
                          "upstream_record_path": path, "upstream_io_index": original["upstream_io_index"]}
            if review is not None:
                replacement = json.dumps(values, indent=4, ensure_ascii=False, allow_nan=False)
                prompt = row["messages"][0]["content"]
                match = list(re.finditer(r"\[INPUT\](.*?)\[/INPUT\]", prompt, re.S))[-1]
                start, end = match.span(1)
                row["messages"][0]["content"] = prompt[:start] + "\n```json\n" + replacement + "\n```\n" + prompt[end:]
                row["original_prompt_sha256"] = original["prompt_sha256"]
                row["prompt_sha256"] = hashlib.sha256(row["messages"][0]["content"].encode("utf-8")).hexdigest()
                row["input_repair"] = "restore_upstream_inputs_as_valid_json"
                repaired.append({"id": case_id, "kind": kind, "before": input_text(original), "after": input_text(row),
                                 "evidence": review["evidence"], "upstream_path": review["upstream_record_path"],
                                 "upstream_io_index": review["upstream_io_index"]})
                if json.loads(input_text(row)) != values:
                    raise ValueError(f"Repair differs from upstream: {case_id}")
            json.loads(input_text(row))  # Fail rather than retain unexpected malformed inputs.
            selected.append(row)
    counts = dict(Counter(r["reason"] for r in excluded))
    repair_counts = dict(Counter(r["kind"] for r in repaired))
    if repair_counts != {"malformed_json": 7, "avatar_string_decode": 27, "avatar_numeric_text": 7}:
        raise ValueError("Unexpected input repair counts")
    if (len(rows), len(selected), len(repaired), counts) != (500, 446, 41, {
        "cruxeval_overlap": 44, "problematic_sample": 10
    }):
        raise ValueError("Selection differs from the reviewed 500-record release")
    final = remove_prompts(selected)
    stdin_programs = sum(not r["entrypoint"] for r in final)
    if stdin_programs != 34:
        raise ValueError("Expected 34 stdin programs among the retained samples")
    return final, {"source_sha256": digest, "input_count": len(rows), "retained_count": len(selected),
                      "excluded_count": len(excluded), "repaired_count": len(repaired),
                      "repair_counts": repair_counts, "stdin_programs": stdin_programs,
                      "exclusion_counts": counts, "excluded": excluded, "repairs": repaired,
                      "limitations": "Screened, not fully validated. Targets unchanged. Seven malformed inputs repaired; 27 Avatar strings decoded and seven numeric stdin values verified."}



def build_re2(data_dir=ROOT / "data"):
    """Write full and final filtered CSVs; return final rows and review report."""
    data_dir = Path(data_dir)
    final, report = prepare_re2(data_dir)
    raw = load_re2_export(data_dir / "raw/re2/re2_bench_lite_output_prediction_prompts.jsonl")
    write_re2_csv(remove_prompts(raw), data_dir / "raw/re2/re2_full.csv")
    write_re2_csv(final, data_dir / "raw/re2/re2_filtered.csv")
    output = data_dir / "processed/re2"
    output.mkdir(parents=True, exist_ok=True)
    (output / "re2_filtered.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final), encoding="utf-8")
    (output / "exclusions.json").write_text(json.dumps(report["excluded"], indent=2) + "\n", encoding="utf-8")
    (output / "selection_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return final, report


if __name__ == "__main__":
    rows, report = build_re2()
    print(f"500 samples -> {len(rows)} retained; 54 excluded; 7 malformed inputs repaired; 34 Avatar inputs normalized (7 numeric values unchanged).")
