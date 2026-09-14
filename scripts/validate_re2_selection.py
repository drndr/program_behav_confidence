"""Inventory every RE2 record and retain only cases with successful bounded replay."""

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from behavior_prediction.datasets import load_re2_export, load_cruxeval, sha256_file
from behavior_prediction.re2_inspection import inspect_re2, input_text


def main():
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    source = next(x for x in provenance if "/re2/" in x["path"])
    rows = load_re2_export(ROOT / source["path"])
    crux = load_cruxeval(ROOT / "data/raw/cruxeval/cruxeval.csv")
    checks = {x["id"]: x for x in inspect_re2(rows, {x.id for x in crux}, ROOT / source["source_path"])}
    previous = json.loads((ROOT / "data/re2_invalid_input_review.json").read_text())
    known_invalid = {x["id"]: x for x in previous["cases"]}
    valid_json_review = json.loads((ROOT / "data/re2_valid_json_review.json").read_text(encoding="utf-8"))
    assert valid_json_review["source_sha256"] == sha256_file(ROOT / source["path"])
    known_contradictions = {x["id"]: x for x in valid_json_review["cases"]}
    old_audit = {x["id"]: x for x in json.loads((ROOT / "data/re2_audit_sample.json").read_text())["cases"]}
    reports, selected = [], []
    output_dir = ROOT / "data/processed/re2_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    worker = ROOT / "src/behavior_prediction/re2_replay.py"
    for index, row in enumerate(rows):
        flag = checks[row["id"]]
        report = {"id": row["id"], "source": row["source_dataset"], "group": row["complexity_group"],
                  "status": "unresolved", "reason": None, "flags": [], "repairs": []}
        reports.append(report)
        if flag["cruxeval_source"]:
            report.update(status="excluded", reason="cruxeval_overlap")
            continue
        if flag["unexposed_address"]:
            report.update(status="excluded", reason="unexposed_reference_address")
            continue
        if row["id"] in old_audit:
            report["earlier_review"] = old_audit[row["id"]]
        if row["id"] in known_contradictions:
            report.update(status="excluded", reason="valid_json_prompt_reference_contradiction",
                          evidence=known_contradictions[row["id"]])
            continue
        if flag["invalid_input_json"]:
            review = known_invalid[row["id"]]
            report.update(status="excluded" if review["recommendation"] == "exclude" else "unresolved",
                          reason="confirmed_missing_state" if review["recommendation"] == "exclude" else "input_repair_pending",
                          evidence=review["evidence"])
            continue
        inputs = json.loads(flag["input_text"])
        prompt = row["messages"][0]["content"]
        body = prompt.rsplit("[/EXAMPLE]", 1)[-1].split("Functions called during the execution:")[0]
        blocks = re.findall(r"\[PYTHON\](.*?)\[/PYTHON\]", body, re.S)
        if not blocks:
            report["reason"] = "code_extraction_failed"
            continue
        code = blocks[-1].strip()
        report["code_sha256"] = hashlib.sha256(code.encode()).hexdigest()
        report["input_sha256"] = hashlib.sha256(flag["input_text"].encode()).hexdigest()
        # Signals trigger review; they do not alone prove that required state is missing.
        serialized = json.dumps(inputs)
        if re.search(r"array\(|masked_array\(|<[^>]+>|\.\.\.", serialized):
            report["flags"].append("object_or_abbreviated_serialization")
        if "null" in serialized:
            report["flags"].append("null_input_values")
        try:
            tree = ast.parse(code)
            self_reads = sorted({n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                                 and isinstance(n.value, ast.Name) and n.value.id == "self" and isinstance(n.ctx, ast.Load)})
            report["self_attributes_not_explicitly_supplied"] = sorted(set(self_reads) - set(inputs.get("self", {})))
        except (SyntaxError, TypeError) as error:
            report.update(reason="source_or_input_not_supported", detail=str(error)[:300])
            continue
        match = re.search(r"What will be the output of\s+`([^`]+)`", prompt.rsplit("[/PYTHON]", 1)[-1])
        function_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        entrypoint = match.group(1).split('.')[-1] if match else (row["id"].split('.')[-1] if row["source_dataset"] == "Classeval" else (function_names[0] if len(function_names) == 1 else None))
        payload = {"code": code, "inputs": inputs, "entrypoint": entrypoint, "source": row["source_dataset"]}
        repaired = copy.deepcopy(row)
        if row["source_dataset"] == "Avatar":
            if not flag["double_encoded_stdin"]:
                report["reason"] = "stdin_encoding_not_confirmed"
                continue
            payload["stdin"] = flag["upstream_stdin"]
            if not isinstance(payload["stdin"], str):
                report["reason"] = "non_string_stdin"
                continue
            new_input = {"input": payload["stdin"]}
            matches = list(re.finditer(r"\[INPUT\](.*?)\[/INPUT\]", prompt, re.S))
            start, end = matches[-1].span(1)
            repaired["messages"][0]["content"] = prompt[:start] + "\n```json\n" + json.dumps(new_input, indent=4, ensure_ascii=False) + "\n```\n" + prompt[end:]
            report["repairs"].append("decode_avatar_stdin_once_verified_against_upstream")
        environment = {key: value for key, value in os.environ.items() if key in {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH"}}
        try:
            process = subprocess.run([sys.executable, "-I", "-S", str(worker)], input=json.dumps(payload),
                                     text=True, encoding="utf-8", capture_output=True, timeout=4, cwd=output_dir, env=environment)
            result = json.loads(process.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            report.update(reason="replay_timeout_or_protocol_failure", detail=type(error).__name__)
            continue
        report["replay"] = result
        if result["status"] != "replayed":
            report["reason"] = result["reason"]
            continue
        target = row["target_value"]
        if row["source_dataset"] == "Avatar":
            observed = result["stdout"]
            # Standard terminal newline is not part of RE2's recorded return.
            value = observed[:-1] if observed.endswith("\n") else observed
            if isinstance(target, str):
                matched = value == target
                policy = "stdout_exact_except_one_terminal_newline"
            elif type(target) in {int, float}:
                try:
                    value = int(value) if type(target) is int else float(value)
                    matched = value == target
                except ValueError:
                    matched = False
                policy = "numeric_stdout_conversion_to_recorded_target_type"
            else:
                matched, policy = False, "unsupported_stdout_target"
        else:
            value = result["value"]
            matched = json.dumps(value, sort_keys=True) == json.dumps(target, sort_keys=True)
            policy = "json_value_equality_tuple_to_list_only"
        report["comparison_policy"] = policy
        if report["flags"] and "object_or_abbreviated_serialization" in report["flags"]:
            report["reason"] = "object_serialization_requires_semantic_review"
        elif matched:
            report.update(status="replay_supported", reason="reference_reproduced_from_prompt_inputs")
            repaired["validation"] = {"status": report["status"], "comparison_policy": policy, "repairs": report["repairs"]}
            if report["repairs"]:
                repaired["validation"]["original_prompt_sha256"] = row["prompt_sha256"]
                repaired["prompt_sha256"] = hashlib.sha256(repaired["messages"][0]["content"].encode()).hexdigest()
            selected.append(repaired)
        else:
            report["reason"] = "reference_mismatch_requires_review"
        if index % 50 == 0:
            print(f"Inspected {index + 1}/500", flush=True)
    summary = {"scope": "All 500 records screened; bounded replay only for supported self-contained Python",
               "source_sha256": sha256_file(ROOT / source["path"]), "status_counts": dict(Counter(x["status"] for x in reports)),
               "reason_counts": dict(Counter(x["reason"] for x in reports)),
               "selected_source_counts": dict(Counter(x["source_dataset"] for x in selected)),
               "selected_group_counts": dict(Counter(x["complexity_group"] for x in selected)),
               "limitations": "Replay-supported is not full historical-environment certification. Unresolved records are not declared defective. Selection favors self-contained executable snippets; not representative of full RE2."}
    (output_dir / "case_reviews.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "replay_supported.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
