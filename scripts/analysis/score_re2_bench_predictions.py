"""Safely score local or released RE2-Bench output predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from re2_bench_scoring import (
    parse_re2_response,
    preparsed_result,
    score_re2_output,
    unwrap_output,
)
from reliability_metrics import summarize_censoring
from utils import read_jsonl, wilson_interval, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--released-validation-csv", type=Path)
    parser.add_argument("--output-items", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    return parser.parse_args()


def load_prompt_map(path: Path) -> dict[str, dict[str, Any]]:
    prompts = {str(row["id"]): row for row in read_jsonl(path)}
    if not prompts:
        raise RuntimeError(f"No prompt rows found in {path}")
    return prompts


def load_released_validation(
    path: Path | None, prompts: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    case_to_id = {
        (str(row["source_dataset"]), str(row["upstream_case_name"])): sample_id
        for sample_id, row in prompts.items()
    }
    validation = {}
    with path.open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            key = (str(row["benchmark"]), str(row["case_name"]))
            sample_id = case_to_id.get(key)
            if sample_id is None:
                raise RuntimeError(f"Validation row has no prompt match: {key}")
            validation[sample_id] = {
                "released_rs": int(float(row["rs"])),
                "released_partial_rs": float(row["partial_rs"]),
            }
    return validation


def score_rows(
    prompts: dict[str, dict[str, Any]],
    predictions_path: Path,
    released_validation: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    released_validation = released_validation or {}
    rows = []
    seen = set()
    for prediction_row in read_jsonl(predictions_path):
        sample_id = str(prediction_row.get("id") or prediction_row.get("problem_id") or "")
        if not sample_id or sample_id in seen:
            raise RuntimeError(f"Missing or duplicate prediction ID: {sample_id!r}")
        seen.add(sample_id)
        if sample_id not in prompts:
            raise RuntimeError(f"Prediction ID is absent from prompts: {sample_id}")
        prompt = prompts[sample_id]
        if "response" in prediction_row:
            parse = parse_re2_response(prediction_row.get("response"))
        else:
            parse = preparsed_result(prediction_row.get("prediction"))
        score = score_re2_output(
            parse,
            prompt.get("target_value"),
            finish_reason=prediction_row.get("finish_reason"),
            native_finish_reason=prediction_row.get("native_finish_reason"),
        )
        rows.append(
            {
                "id": sample_id,
                "dataset": "re2_bench_lite",
                "source_dataset": prompt.get("source_dataset"),
                "complexity_group": prompt.get("complexity_group"),
                "model": prediction_row.get("model"),
                "model_revision": prediction_row.get("model_revision"),
                "response_model": prediction_row.get("response_model"),
                "provider": prediction_row.get("provider"),
                "config_hash": prediction_row.get("config_hash"),
                "arm_id": prediction_row.get("arm_id"),
                "replicate_seed": prediction_row.get("replicate_seed"),
                "reasoning_effort": prediction_row.get("reasoning_effort"),
                "max_tokens": prediction_row.get("max_tokens"),
                "error": prediction_row.get("error"),
                "prompt_sha256": prompt.get("prompt_sha256"),
                "response": prediction_row.get("response"),
                "target_value": prompt.get("target_value"),
                "parsed_prediction_value": unwrap_output(parse.value) if parse.parsed else None,
                **score.to_dict(),
                **released_validation.get(sample_id, {}),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("complexity_group"))].append(row)

    def one_group(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        policies = {
            field: summarize_censoring(group_rows, correct_field=field).to_dict()
            for field in (
                "strict_json_typed_correct",
                "repaired_json_typed_correct",
                "safe_compatibility_correct",
            )
        }
        for field, policy in policies.items():
            lower, upper = wilson_interval(policy["correct_total"], policy["total"])
            policy["wilson_95"] = [lower, upper]
        summary = {
            "count": len(group_rows),
            "parse_status": dict(Counter(row["parse_status"] for row in group_rows)),
            "parse_mode": dict(Counter(str(row["parse_mode"]) for row in group_rows)),
            "policies": policies,
            "scoring_policy_disagreements": sum(
                row["repaired_json_typed_correct"] != row["safe_compatibility_correct"]
                for row in group_rows
            ),
        }
        if group_rows and all("released_rs" in row for row in group_rows):
            summary["released_validation"] = {
                "correct": sum(row["released_rs"] for row in group_rows),
                "accuracy": sum(row["released_rs"] for row in group_rows) / len(group_rows),
                "agreement_with_safe_compatibility": sum(
                    bool(row["released_rs"]) == row["safe_compatibility_correct"]
                    for row in group_rows
                )
                / len(group_rows),
            }
        return summary

    return {
        "total_rows": len(rows),
        "overall": one_group(rows),
        "by_complexity_group": {
            group: one_group(group_rows) for group, group_rows in sorted(grouped.items())
        },
        "by_source_dataset": {
            source: one_group([row for row in rows if row.get("source_dataset") == source])
            for source in sorted({str(row.get("source_dataset")) for row in rows})
        },
        "by_complexity_and_source": {
            f"{group}:{source}": one_group(
                [
                    row
                    for row in rows
                    if row.get("complexity_group") == group
                    and row.get("source_dataset") == source
                ]
            )
            for group, source in sorted(
                {
                    (str(row.get("complexity_group")), str(row.get("source_dataset")))
                    for row in rows
                }
            )
        },
    }


def main() -> None:
    args = parse_args()
    prompts = load_prompt_map(args.prompts)
    released_validation = load_released_validation(args.released_validation_csv, prompts)
    rows = score_rows(prompts, args.predictions, released_validation)
    summary = summarize_rows(rows)
    write_jsonl(args.output_items, rows, sort_keys=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Scored rows: {len(rows)}")
    for group, group_summary in summary["by_complexity_group"].items():
        policies = group_summary["policies"]
        print(
            f"{group}: strict={100 * policies['strict_json_typed_correct']['system_accuracy']:.2f}% "
            f"repaired={100 * policies['repaired_json_typed_correct']['system_accuracy']:.2f}% "
            f"compatible={100 * policies['safe_compatibility_correct']['system_accuracy']:.2f}%"
        )
    print(f"Items:   {args.output_items.resolve()}")
    print(f"Summary: {args.output_summary.resolve()}")


if __name__ == "__main__":
    main()
