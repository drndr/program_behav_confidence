"""Compare code-aware and candidate-only confidence on paired CRUXEval items."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from analyze_cruxeval_calibration_pilot import (  # noqa: E402
    brier_score,
    equal_mass_ece,
    metric_bundle,
    parse_confidence,
    percentile_interval,
    risk_coverage,
    roc_auc,
)
from audit_cruxeval_calibration_internal_validity import (  # noqa: E402
    CONDITIONS,
    adjusted_confidence_audit,
    fit_logistic,
    load_joined_items,
    standardize,
    stratified_auc,
)


DEFAULT_BOOTSTRAPS = 2000
DEFAULT_SEED = 20260817


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "research_design/2026-08-17_cruxeval_prediction_only_confidence_control_analysis.json"
        ),
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=Path(
            "research_design/2026-08-17_cruxeval_prediction_only_confidence_control_results.txt"
        ),
    )
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with resolve_path(path).open(encoding="utf-8") as input_file:
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
        raise RuntimeError(f"No rows found in {path}")
    return rows


def index_unique(path: Path) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in indexed:
            raise ValueError(f"Missing or duplicate ID in {path}: {sample_id!r}")
        indexed[sample_id] = row
    return indexed


def control_paths(condition: str) -> dict[str, Path]:
    return {
        "prompts": Path(
            f"data/prompts/cruxeval/cruxeval_prediction_only_confidence_{condition}_v1.jsonl"
        ),
        "results": Path(
            "results/"
            f"cruxeval_prediction_only_confidence_gpt_5_4_nano_{condition}_predictions.jsonl"
        ),
        "metadata": Path(
            "results/"
            f"cruxeval_prediction_only_confidence_gpt_5_4_nano_{condition}_predictions_metadata.json"
        ),
    }


def load_control_confidence() -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    confidence: dict[str, dict[str, float]] = {}
    failures: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, Any] = {}
    for condition in CONDITIONS:
        paths = control_paths(condition)
        prompts = index_unique(paths["prompts"])
        results = index_unique(paths["results"])
        if set(prompts) != set(results):
            raise ValueError(f"{condition} control result IDs do not match prompts")
        condition_confidence = {}
        condition_failures = []
        for sample_id, result in results.items():
            try:
                condition_confidence[sample_id] = parse_confidence(
                    str(result.get("response", result.get("prediction", "")))
                )
            except ValueError as exc:
                condition_failures.append(
                    {
                        "id": sample_id,
                        "split": prompts[sample_id]["split"],
                        "error": str(exc),
                        "finish_reason": result.get("finish_reason"),
                        "native_finish_reason": result.get("native_finish_reason"),
                        "generated_token_count": result.get("generated_token_count"),
                        "reasoning_token_count": result.get("reasoning_token_count"),
                    }
                )
        confidence[condition] = condition_confidence
        failures[condition] = sorted(condition_failures, key=lambda row: row["id"])
        metadata[condition] = json.loads(
            resolve_path(paths["metadata"]).read_text(encoding="utf-8")
        )
    return confidence, {
        "parse_failures": failures,
        "run_metadata": metadata,
        "paths": {
            condition: {
                label: str(resolve_path(path))
                for label, path in control_paths(condition).items()
            }
            for condition in CONDITIONS
        },
    }


def build_paired_rows() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    code_aware_items, code_aware_metadata = load_joined_items()
    control_confidence, control_metadata = load_control_confidence()
    code_by_id = {
        condition: {row["id"]: row for row in rows}
        for condition, rows in code_aware_items.items()
    }
    common_ids = set(code_by_id["none"])
    for condition in CONDITIONS:
        common_ids &= set(code_by_id[condition])
        common_ids &= set(control_confidence[condition])

    paired: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        rows = []
        for sample_id in sorted(common_ids):
            source = code_by_id[condition][sample_id]
            rows.append(
                {
                    **source,
                    "code_aware_confidence": float(source["confidence"]),
                    "prediction_only_confidence": float(
                        control_confidence[condition][sample_id]
                    ),
                }
            )
        paired[condition] = rows

    for split in ("calibration", "test"):
        ids = [
            {row["id"] for row in paired[condition] if row["split"] == split}
            for condition in CONDITIONS
        ]
        if ids[0] != ids[1]:
            raise ValueError(f"Paired IDs differ across reasoning conditions for {split}")

    return paired, {
        "paired_complete_cases": len(common_ids),
        "paired_split_sizes": {
            split: sum(row["split"] == split for row in paired["none"])
            for split in ("calibration", "test")
        },
        "code_aware": code_aware_metadata,
        "prediction_only": control_metadata,
    }


def with_confidence(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [{**row, "confidence": float(row[field])} for row in rows]


def information_condition_metrics(
    rows: list[dict[str, Any]], confidence_field: str
) -> dict[str, Any]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    test = [row for row in rows if row["split"] == "test"]
    y_calibration = np.asarray(
        [row["semantic_correct"] for row in calibration], dtype=int
    )
    y_test = np.asarray([row["semantic_correct"] for row in test], dtype=int)
    probability = np.asarray([row[confidence_field] for row in test], dtype=float)
    baseline = float(np.mean(y_calibration))
    metric = metric_bundle(y_test, probability, baseline)
    prepared = with_confidence(rows, confidence_field)
    test_prepared = [row for row in prepared if row["split"] == "test"]
    return {
        "metrics": metric,
        "target_type_stratified_auc": stratified_auc(
            test_prepared, lambda row: row["target_type"]
        ),
        "adjusted_confidence": adjusted_confidence_audit(prepared),
        "risk_coverage": risk_coverage(y_test, probability),
        "unique_confidence_values": int(len(np.unique(probability))),
    }


def paired_bootstrap(
    rows: list[dict[str, Any]], repetitions: int, seed: int
) -> dict[str, Any]:
    test = [row for row in rows if row["split"] == "test"]
    calibration = [row for row in rows if row["split"] == "calibration"]
    baseline = float(np.mean([row["semantic_correct"] for row in calibration]))
    y = np.asarray([row["semantic_correct"] for row in test], dtype=int)
    aware = np.asarray([row["code_aware_confidence"] for row in test], dtype=float)
    control = np.asarray(
        [row["prediction_only_confidence"] for row in test], dtype=float
    )
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = defaultdict(list)
    for _ in range(repetitions):
        indices = rng.integers(0, len(test), size=len(test))
        sampled_y = y[indices]
        sampled_aware = aware[indices]
        sampled_control = control[indices]
        aware_auc = roc_auc(sampled_y, sampled_aware)
        control_auc = roc_auc(sampled_y, sampled_control)
        if aware_auc is not None and control_auc is not None:
            collected["auroc_code_aware_minus_prediction_only"].append(
                float(aware_auc - control_auc)
            )
        collected["brier_prediction_only_minus_code_aware"].append(
            brier_score(sampled_y, sampled_control)
            - brier_score(sampled_y, sampled_aware)
        )
        collected["ece_prediction_only_minus_code_aware"].append(
            equal_mass_ece(sampled_y, sampled_control)
            - equal_mass_ece(sampled_y, sampled_aware)
        )
        baseline_vector = np.full(len(sampled_y), baseline)
        baseline_brier = brier_score(sampled_y, baseline_vector)
        if baseline_brier > 0:
            aware_skill = 1.0 - brier_score(sampled_y, sampled_aware) / baseline_brier
            control_skill = 1.0 - brier_score(sampled_y, sampled_control) / baseline_brier
            collected["brier_skill_code_aware_minus_prediction_only"].append(
                aware_skill - control_skill
            )
    return {key: percentile_interval(values) for key, values in collected.items()}


def comparison_point_estimates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    test = [row for row in rows if row["split"] == "test"]
    y = np.asarray([row["semantic_correct"] for row in test], dtype=int)
    aware = np.asarray([row["code_aware_confidence"] for row in test], dtype=float)
    control = np.asarray(
        [row["prediction_only_confidence"] for row in test], dtype=float
    )
    aware_auc = roc_auc(y, aware)
    control_auc = roc_auc(y, control)
    if len(np.unique(aware)) < 2 or len(np.unique(control)) < 2:
        correlation_rho = None
        correlation_p_value = None
    else:
        correlation = spearmanr(aware, control)
        correlation_rho = float(correlation.statistic)
        correlation_p_value = float(correlation.pvalue)
    return {
        "auroc_code_aware_minus_prediction_only": (
            None if aware_auc is None or control_auc is None else aware_auc - control_auc
        ),
        "brier_prediction_only_minus_code_aware": (
            brier_score(y, control) - brier_score(y, aware)
        ),
        "ece_prediction_only_minus_code_aware": (
            equal_mass_ece(y, control) - equal_mass_ece(y, aware)
        ),
        "mean_confidence_code_aware_minus_prediction_only": float(
            np.mean(aware - control)
        ),
        "mean_absolute_confidence_difference": float(np.mean(np.abs(aware - control))),
        "confidence_spearman_rho": correlation_rho,
        "confidence_spearman_p_value": correlation_p_value,
    }


def incremental_confidence_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    test = [row for row in rows if row["split"] == "test"]
    outcomes_by_type: dict[str, set[bool]] = defaultdict(set)
    for row in test:
        outcomes_by_type[row["target_type"]].add(row["semantic_correct"])
    estimable_types = sorted(
        target_type
        for target_type, outcomes in outcomes_by_type.items()
        if len(outcomes) == 2
    )
    selected = [row for row in test if row["target_type"] in estimable_types]
    reference_type = max(
        estimable_types,
        key=lambda target_type: sum(row["target_type"] == target_type for row in selected),
    )
    type_columns = [value for value in estimable_types if value != reference_type]
    code_length = standardize(
        np.log1p(np.asarray([row["code_chars"] for row in selected], dtype=float))
    )
    output_length = standardize(
        np.log1p(np.asarray([row["target_chars"] for row in selected], dtype=float))
    )
    covariates = np.asarray(
        [
            [1.0, code_length[index], output_length[index]]
            + [float(row["target_type"] == value) for value in type_columns]
            for index, row in enumerate(selected)
        ],
        dtype=float,
    )
    aware = np.asarray(
        [10.0 * (row["code_aware_confidence"] - 0.5) for row in selected],
        dtype=float,
    )
    control = np.asarray(
        [10.0 * (row["prediction_only_confidence"] - 0.5) for row in selected],
        dtype=float,
    )
    y = np.asarray([row["semantic_correct"] for row in selected], dtype=float)

    def addition_test(existing: np.ndarray, added: np.ndarray) -> dict[str, Any]:
        reduced_design = np.column_stack((covariates, existing))
        full_design = np.column_stack((reduced_design, added))
        reduced = fit_logistic(reduced_design, y)
        full = fit_logistic(full_design, y)
        statistic = max(
            0.0, 2.0 * (full["log_likelihood"] - reduced["log_likelihood"])
        )
        coefficient = float(full["beta"][-1])
        standard_error = float(full["standard_errors"][-1])
        return {
            "coefficient_per_10_points": coefficient,
            "standard_error": standard_error,
            "odds_ratio_per_10_points": math.exp(coefficient),
            "likelihood_ratio_statistic": statistic,
            "likelihood_ratio_p_value": float(chi2.sf(statistic, 1)),
            "reduced_fit_success": reduced["success"],
            "full_fit_success": full["success"],
        }

    return {
        "n": len(selected),
        "estimable_target_types": estimable_types,
        "reference_target_type": reference_type,
        "code_aware_added_after_prediction_only": addition_test(control, aware),
        "prediction_only_added_after_code_aware": addition_test(aware, control),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    data = summary["data"]
    lines = [
        "CRUXEval prediction-only confidence negative control",
        "=" * 52,
        "",
        (
            f"Paired complete cases across both information and reasoning conditions: "
            f"{data['paired_complete_cases']} "
            f"(calibration={data['paired_split_sizes']['calibration']}, "
            f"test={data['paired_split_sizes']['test']})."
        ),
        "",
    ]
    for condition in CONDITIONS:
        result = summary["conditions"][condition]
        aware = result["code_aware"]["metrics"]
        control = result["prediction_only"]["metrics"]
        comparison = result["comparison"]
        intervals = result["paired_bootstrap_95_ci"]
        auc_interval = intervals["auroc_code_aware_minus_prediction_only"]
        brier_interval = intervals["brier_prediction_only_minus_code_aware"]
        lines.extend(
            [
                f"{condition.upper()} REASONING",
                "-" * (len(condition) + 10),
                (
                    f"Code-aware:      AUROC={fmt(aware['auroc'])}, "
                    f"Brier={fmt(aware['brier_score'])}, "
                    f"Brier skill={fmt(aware['brier_skill_score'])}, "
                    f"ECE={fmt(aware['ece_equal_mass_5'])}."
                ),
                (
                    f"Prediction-only: AUROC={fmt(control['auroc'])}, "
                    f"Brier={fmt(control['brier_score'])}, "
                    f"Brier skill={fmt(control['brier_skill_score'])}, "
                    f"ECE={fmt(control['ece_equal_mass_5'])}."
                ),
                (
                    f"AUROC gain from showing code/input: "
                    f"{fmt(comparison['auroc_code_aware_minus_prediction_only'])} "
                    f"(95% CI {fmt(auc_interval[0])}-{fmt(auc_interval[1])})."
                ),
                (
                    f"Brier improvement from showing code/input: "
                    f"{fmt(comparison['brier_prediction_only_minus_code_aware'])} "
                    f"(95% CI {fmt(brier_interval[0])}-{fmt(brier_interval[1])}; "
                    "positive favors code-aware)."
                ),
                (
                    "Type-stratified AUROC, code-aware/prediction-only: "
                    f"{fmt(result['code_aware']['target_type_stratified_auc']['auroc'])}/"
                    f"{fmt(result['prediction_only']['target_type_stratified_auc']['auroc'])}."
                ),
                (
                    "Correlation between the two confidence scores: "
                    f"rho={fmt(comparison['confidence_spearman_rho'])}."
                ),
            ]
        )
        incremental = result["incremental_confidence"]
        added_aware = incremental["code_aware_added_after_prediction_only"]
        added_control = incremental["prediction_only_added_after_code_aware"]
        lines.append(
            "Adjusted incremental tests: code-aware after prediction-only "
            f"LR p={added_aware['likelihood_ratio_p_value']:.3g}; "
            "prediction-only after code-aware "
            f"LR p={added_control['likelihood_ratio_p_value']:.3g}."
        )
        lines.append("")

    lines.extend(["Run completeness and cost", "-------------------------"])
    for condition in CONDITIONS:
        metadata = data["prediction_only"]["run_metadata"][condition]
        failures = data["prediction_only"]["parse_failures"][condition]
        lines.append(
            f"{condition}: completed={metadata.get('completed_rows')}, "
            f"truncated={metadata.get('truncated_rows')}, "
            f"unparsed confidence={len(failures)}, "
            f"cost=${float(metadata.get('reported_cost_usd', 0)):.4f}."
        )

    lines.extend(["", "Interpretation", "--------------", summary["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paired, data = build_paired_rows()
    conditions = {}
    for index, condition in enumerate(CONDITIONS):
        rows = paired[condition]
        conditions[condition] = {
            "code_aware": information_condition_metrics(
                rows, "code_aware_confidence"
            ),
            "prediction_only": information_condition_metrics(
                rows, "prediction_only_confidence"
            ),
            "comparison": comparison_point_estimates(rows),
            "paired_bootstrap_95_ci": paired_bootstrap(
                rows, args.bootstraps, args.seed + index
            ),
            "incremental_confidence": incremental_confidence_audit(rows),
        }

    low_gain = conditions["low"]["comparison"][
        "auroc_code_aware_minus_prediction_only"
    ]
    low_interval = conditions["low"]["paired_bootstrap_95_ci"][
        "auroc_code_aware_minus_prediction_only"
    ]
    if low_gain is not None and low_interval is not None and low_interval[0] > 0:
        interpretation = (
            "For low reasoning, access to the program and input produces significantly greater "
            "error discrimination than candidate surface cues alone. The reflective-confidence "
            "signal therefore cannot be reduced to output plausibility or output-type base rates."
        )
    elif low_gain is not None and abs(low_gain) < 0.03:
        interpretation = (
            "For low reasoning, candidate-only and code-aware confidence discriminate errors "
            "similarly. Most of the apparent reflective-confidence signal may therefore come from "
            "output surface cues or condition base rates rather than program verification."
        )
    else:
        interpretation = (
            "The pilot does not cleanly separate code-aware verification from candidate surface "
            "cues. A larger or differently constructed control is needed before making that claim."
        )

    summary = {
        "analysis": "cruxeval_prediction_only_confidence_control_v1",
        "primary_correctness": "relaxed_correct semantic Python value equivalence",
        "bootstraps": args.bootstraps,
        "bootstrap_seed": args.seed,
        "data": data,
        "conditions": conditions,
        "interpretation": interpretation,
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
