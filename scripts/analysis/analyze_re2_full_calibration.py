"""Analyze the full RE2-Bench paired confidence experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_cruxeval_calibration_pilot import (  # noqa: E402
    metric_bundle,
    parse_confidence,
    percentile_interval,
    risk_coverage,
    roc_auc,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONDITIONS = ("none", "low")
INFORMATION_CONDITIONS = ("response_only", "code_aware")
DEFAULT_BOOTSTRAPS = 5000
DEFAULT_SEED = 20260819


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("tmp"))
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("research_design/2026-08-19_re2_full_calibration_analysis.json"),
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=Path("research_design/2026-08-19_re2_full_calibration_results.txt"),
    )
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No records found in {path}")
    return rows


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in indexed:
            raise ValueError(f"Missing or duplicate ID in {label}: {sample_id!r}")
        indexed[sample_id] = row
    return indexed


def load_condition(
    prompt_path: Path,
    result_path: Path,
    source_condition: str,
    information_condition: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    prompts = index_unique(read_jsonl(prompt_path), str(prompt_path))
    results = index_unique(read_jsonl(result_path), str(result_path))
    if set(prompts) != set(results) or len(prompts) != 500:
        raise ValueError(
            f"Expected 500 matching prompt/result IDs for "
            f"{source_condition}/{information_condition}"
        )
    items: dict[str, dict[str, Any]] = {}
    parse_failures = []
    for sample_id in sorted(prompts):
        prompt = prompts[sample_id]
        result = results[sample_id]
        if prompt.get("source_condition") != source_condition:
            raise ValueError(f"Source-condition mismatch for {sample_id}")
        if prompt.get("information_condition") != information_condition:
            raise ValueError(f"Information-condition mismatch for {sample_id}")
        response = str(result.get("response", result.get("prediction", "")))
        parse_error = None
        try:
            parsed_confidence = parse_confidence(response)
        except ValueError as exc:
            parsed_confidence = None
            parse_error = str(exc)
            parse_failures.append(
                {
                    "id": sample_id,
                    "error": parse_error,
                    "finish_reason": result.get("finish_reason"),
                    "native_finish_reason": result.get("native_finish_reason"),
                    "generated_token_count": result.get("generated_token_count"),
                    "reasoning_token_count": result.get("reasoning_token_count"),
                }
            )
        items[sample_id] = {
            "id": sample_id,
            "source_condition": source_condition,
            "information_condition": information_condition,
            "correct": bool(prompt["safe_compatibility_correct"]),
            "strict_correct": bool(prompt["strict_json_typed_correct"]),
            "repaired_correct": bool(prompt["repaired_json_typed_correct"]),
            "confidence": parsed_confidence,
            # Operational policy: an unusable confidence response cannot authorize use.
            "operational_confidence": 0.0 if parsed_confidence is None else parsed_confidence,
            "confidence_parse_error": parse_error,
            "is_truncated": bool(
                result.get("finish_reason") == "length"
                or result.get("native_finish_reason")
                in {"max_tokens", "max_output_tokens"}
            ),
            "source_dataset": str(prompt["source_dataset"]),
            "complexity_group": str(prompt["complexity_group"]),
            "target_type": str(prompt["target_type"]),
            "matched_132_subset": bool(prompt["matched_132_subset"]),
        }
    return items, parse_failures


def arrays(
    items: dict[str, dict[str, Any]],
    sample_ids: list[str],
    confidence_field: str = "operational_confidence",
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([items[sample_id]["correct"] for sample_id in sample_ids], dtype=int),
        np.asarray([items[sample_id][confidence_field] for sample_id in sample_ids], dtype=float),
    )


def summarize(
    items: dict[str, dict[str, Any]], sample_ids: list[str], confidence_field: str
) -> dict[str, Any]:
    y, probability = arrays(items, sample_ids, confidence_field)
    metrics = metric_bundle(y, probability, float(np.mean(y)))
    metrics["risk_coverage"] = risk_coverage(y, probability)
    return metrics


def paired_information_bootstrap(
    response_items: dict[str, dict[str, Any]],
    code_items: dict[str, dict[str, Any]],
    sample_ids: list[str],
    repetitions: int,
    seed: int,
    confidence_field: str = "operational_confidence",
) -> dict[str, Any]:
    response_y, response_probability = arrays(response_items, sample_ids, confidence_field)
    code_y, code_probability = arrays(code_items, sample_ids, confidence_field)
    if not np.array_equal(response_y, code_y):
        raise ValueError("Paired information conditions have different correctness labels")
    response_auc = roc_auc(response_y, response_probability)
    code_auc = roc_auc(code_y, code_probability)
    if response_auc is None or code_auc is None:
        raise ValueError("Both correctness classes are required for AUROC")
    response_brier = float(np.mean((response_probability - response_y) ** 2))
    code_brier = float(np.mean((code_probability - code_y) ** 2))
    rng = np.random.default_rng(seed)
    auc_differences = []
    brier_improvements = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(sample_ids), size=len(sample_ids))
        sampled_y = response_y[indices]
        response_sample = response_probability[indices]
        code_sample = code_probability[indices]
        sampled_response_auc = roc_auc(sampled_y, response_sample)
        sampled_code_auc = roc_auc(sampled_y, code_sample)
        if sampled_response_auc is not None and sampled_code_auc is not None:
            auc_differences.append(sampled_code_auc - sampled_response_auc)
        brier_improvements.append(
            float(
                np.mean((response_sample - sampled_y) ** 2)
                - np.mean((code_sample - sampled_y) ** 2)
            )
        )
    return {
        "n": len(sample_ids),
        "response_only_auroc": response_auc,
        "code_aware_auroc": code_auc,
        "code_aware_minus_response_only_auroc": code_auc - response_auc,
        "auroc_difference_95_ci": percentile_interval(auc_differences),
        "response_only_brier": response_brier,
        "code_aware_brier": code_brier,
        "response_only_minus_code_aware_brier": response_brier - code_brier,
        "brier_improvement_95_ci": percentile_interval(brier_improvements),
    }


def paired_accuracy_effect(
    none_items: dict[str, dict[str, Any]],
    low_items: dict[str, dict[str, Any]],
    sample_ids: list[str],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    none_y, _ = arrays(none_items, sample_ids)
    low_y, _ = arrays(low_items, sample_ids)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(sample_ids), size=len(sample_ids))
        differences.append(float(np.mean(low_y[indices] - none_y[indices])))
    patterns = Counter(zip(none_y.tolist(), low_y.tolist()))
    return {
        "n": len(sample_ids),
        "none_accuracy": float(np.mean(none_y)),
        "low_accuracy": float(np.mean(low_y)),
        "low_minus_none": float(np.mean(low_y - none_y)),
        "low_minus_none_95_ci": percentile_interval(differences),
        "paired_correctness_patterns": {
            f"none_{none}_low_{low}": patterns[(none, low)]
            for none in (0, 1)
            for low in (0, 1)
        },
    }


def subgroup_summaries(items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("complexity_group", "source_dataset", "target_type"):
        output[field] = {}
        values = sorted({str(row[field]) for row in items.values()})
        for value in values:
            sample_ids = sorted(
                sample_id
                for sample_id, row in items.items()
                if str(row[field]) == value
            )
            output[field][value] = summarize(items, sample_ids, "operational_confidence")
    return output


def fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "Full RE2-Bench calibration experiment",
        "=" * 36,
        "",
        "Correctness target: safe compatibility system success.",
        "Confidence-judge reasoning effort: low in all information conditions.",
        "Unparseable confidence responses receive operational confidence zero; complete-case sensitivity is also reported.",
        "",
        "Source output prediction",
        "------------------------",
    ]
    source = summary["source_accuracy_effect"]
    interval = source["low_minus_none_95_ci"]
    lines.append(
        f"none={source['none_accuracy']:.1%}; low={source['low_accuracy']:.1%}; "
        f"paired delta={source['low_minus_none']:+.1%} "
        f"(95% bootstrap CI {interval[0]:+.1%} to {interval[1]:+.1%})."
    )
    lines.append(f"Paired correctness patterns: {source['paired_correctness_patterns']}")
    lines.extend(["", "Confidence extraction", "---------------------"])
    for source_condition in SOURCE_CONDITIONS:
        for information_condition in INFORMATION_CONDITIONS:
            extraction = summary["confidence_extraction"][source_condition][information_condition]
            lines.append(
                f"{source_condition}/{information_condition}: "
                f"parsed={extraction['parsed']}/500, truncated={extraction['truncated']}."
            )
    lines.extend(["", "Operational raw confidence", "--------------------------"])
    for source_condition in SOURCE_CONDITIONS:
        for information_condition in INFORMATION_CONDITIONS:
            metrics = summary["operational_metrics"][source_condition][information_condition]
            lines.append(
                f"{source_condition}/{information_condition}: "
                f"accuracy={metrics['accuracy']:.3f}, mean_confidence={metrics['mean_confidence']:.3f}, "
                f"gap={metrics['calibration_gap']:+.3f}, AUROC={fmt(metrics['auroc'])}, "
                f"Brier={metrics['brier_score']:.3f}, BrierSkill={metrics['brier_skill_score']:.3f}."
            )
    lines.extend(["", "Primary paired information effect", "---------------------------------"])
    for source_condition in SOURCE_CONDITIONS:
        effect = summary["paired_information_effect"][source_condition]["operational"]
        auc_interval = effect["auroc_difference_95_ci"]
        brier_interval = effect["brier_improvement_95_ci"]
        lines.append(
            f"{source_condition}: AUROC {effect['response_only_auroc']:.3f} -> "
            f"{effect['code_aware_auroc']:.3f}; delta={effect['code_aware_minus_response_only_auroc']:+.3f} "
            f"(95% CI {auc_interval[0]:+.3f} to {auc_interval[1]:+.3f})."
        )
        lines.append(
            f"  Brier {effect['response_only_brier']:.3f} -> {effect['code_aware_brier']:.3f}; "
            f"improvement={effect['response_only_minus_code_aware_brier']:+.3f} "
            f"(95% CI {brier_interval[0]:+.3f} to {brier_interval[1]:+.3f})."
        )
        matched = summary["paired_information_effect"][source_condition]["matched_132"]
        lines.append(
            f"  Matched 132 subset: AUROC delta={matched['code_aware_minus_response_only_auroc']:+.3f}."
        )
    lines.extend(["", "Code-aware selective prediction", "-------------------------------"])
    for source_condition in SOURCE_CONDITIONS:
        rows = summary["operational_metrics"][source_condition]["code_aware"]["risk_coverage"]
        lines.append(f"{source_condition}:")
        for row in rows:
            lines.append(
                f"  actual coverage={row['actual_coverage']:.1%}: accuracy={row['accuracy']:.1%}, "
                f"minimum confidence={row['minimum_confidence']:.2f}."
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstraps <= 0:
        raise ValueError("--bootstraps must be positive")
    input_dir = resolve_path(args.input_dir)
    items: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    failures: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for source_condition in SOURCE_CONDITIONS:
        items[source_condition] = {}
        failures[source_condition] = {}
        for information_condition in INFORMATION_CONDITIONS:
            prompt_path = input_dir / (
                f"re2_bench_full_confidence_{source_condition}_{information_condition}_v1.jsonl"
            )
            result_path = input_dir / (
                f"re2_bench_full_confidence_{source_condition}_{information_condition}_predictions.jsonl"
            )
            loaded, parse_failures = load_condition(
                prompt_path,
                result_path,
                source_condition,
                information_condition,
            )
            items[source_condition][information_condition] = loaded
            failures[source_condition][information_condition] = parse_failures

    sample_ids = sorted(items["none"]["response_only"])
    for source_condition in SOURCE_CONDITIONS:
        for information_condition in INFORMATION_CONDITIONS:
            if sorted(items[source_condition][information_condition]) != sample_ids:
                raise ValueError("All four conditions must contain the same 500 IDs")

    operational_metrics = {
        source_condition: {
            information_condition: summarize(
                items[source_condition][information_condition],
                sample_ids,
                "operational_confidence",
            )
            for information_condition in INFORMATION_CONDITIONS
        }
        for source_condition in SOURCE_CONDITIONS
    }
    paired_effect: dict[str, Any] = {}
    for condition_index, source_condition in enumerate(SOURCE_CONDITIONS):
        response_items = items[source_condition]["response_only"]
        code_items = items[source_condition]["code_aware"]
        complete_ids = [
            sample_id
            for sample_id in sample_ids
            if response_items[sample_id]["confidence"] is not None
            and code_items[sample_id]["confidence"] is not None
        ]
        matched_ids = [
            sample_id
            for sample_id in sample_ids
            if response_items[sample_id]["matched_132_subset"]
        ]
        paired_effect[source_condition] = {
            "operational": paired_information_bootstrap(
                response_items,
                code_items,
                sample_ids,
                args.bootstraps,
                args.seed + condition_index,
            ),
            "complete_case": paired_information_bootstrap(
                response_items,
                code_items,
                complete_ids,
                args.bootstraps,
                args.seed + 10 + condition_index,
                confidence_field="confidence",
            ),
            "matched_132": paired_information_bootstrap(
                response_items,
                code_items,
                matched_ids,
                args.bootstraps,
                args.seed + 20 + condition_index,
            ),
        }

    summary = {
        "analysis": "re2_full_calibration_v1",
        "correctness_target": "safe_compatibility_correct",
        "confidence_failure_policy": (
            "Operational primary analysis assigns confidence zero to an unusable confidence "
            "response; complete-case results are a sensitivity analysis."
        ),
        "bootstraps": args.bootstraps,
        "bootstrap_seed": args.seed,
        "source_accuracy_effect": paired_accuracy_effect(
            items["none"]["response_only"],
            items["low"]["response_only"],
            sample_ids,
            args.bootstraps,
            args.seed + 30,
        ),
        "confidence_extraction": {
            source_condition: {
                information_condition: {
                    "total": 500,
                    "parsed": 500 - len(failures[source_condition][information_condition]),
                    "parse_failures": len(failures[source_condition][information_condition]),
                    "truncated": sum(
                        row["is_truncated"]
                        for row in items[source_condition][information_condition].values()
                    ),
                    "failure_items": failures[source_condition][information_condition],
                }
                for information_condition in INFORMATION_CONDITIONS
            }
            for source_condition in SOURCE_CONDITIONS
        },
        "operational_metrics": operational_metrics,
        "paired_information_effect": paired_effect,
        "code_aware_subgroups": {
            source_condition: subgroup_summaries(items[source_condition]["code_aware"])
            for source_condition in SOURCE_CONDITIONS
        },
    }
    json_output = resolve_path(args.json_output)
    text_output = resolve_path(args.text_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = render_report(summary)
    text_output.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"JSON analysis: {json_output}")
    print(f"Text report: {text_output}")


if __name__ == "__main__":
    main()
