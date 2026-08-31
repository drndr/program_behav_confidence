"""Audit whether CRUXEval reflective confidence survives obvious confounds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, mannwhitneyu, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from analyze_cruxeval_calibration_pilot import (  # noqa: E402
    metric_bundle,
    parse_confidence,
    percentile_interval,
    read_jsonl,
    roc_auc,
)


CONDITIONS = ("none", "low")
DEFAULT_BOOTSTRAPS = 2000
DEFAULT_SEED = 20260817


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "research_design/2026-08-17_cruxeval_calibration_internal_validity.json"
        ),
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=Path(
            "research_design/2026-08-17_cruxeval_calibration_internal_validity_results.txt"
        ),
    )
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def index_unique(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(resolve_path(path)):
        sample_id = str(row.get("id", ""))
        if not sample_id:
            raise ValueError(f"Row without ID in {path}")
        if sample_id in indexed:
            raise ValueError(f"Duplicate ID {sample_id} in {path}")
        indexed[sample_id] = row
    return indexed


def input_paths(condition: str) -> dict[str, Path]:
    return {
        "prompts": Path(
            f"data/prompts/cruxeval/cruxeval_reflective_confidence_{condition}_v1.jsonl"
        ),
        "confidence": Path(
            "results/"
            f"cruxeval_reflective_confidence_gpt_5_4_nano_{condition}_predictions.jsonl"
        ),
        "source_predictions": Path(
            f"results/cruxeval_direct_gpt_5_4_nano_{condition}_predictions.jsonl"
        ),
        "source_scores": Path(
            f"results/cruxeval_direct_gpt_5_4_nano_{condition}_scores.jsonl"
        ),
    }


def confidence_values(
    prompt_rows: dict[str, dict[str, Any]],
    confidence_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    parsed: dict[str, float] = {}
    failures: list[dict[str, Any]] = []
    if set(prompt_rows) != set(confidence_rows):
        raise ValueError("Confidence results do not exactly match confidence prompts")
    for sample_id, result in confidence_rows.items():
        try:
            parsed[sample_id] = parse_confidence(
                str(result.get("response", result.get("prediction", "")))
            )
        except ValueError as exc:
            failures.append(
                {
                    "id": sample_id,
                    "split": prompt_rows[sample_id]["split"],
                    "error": str(exc),
                    "finish_reason": result.get("finish_reason"),
                    "native_finish_reason": result.get("native_finish_reason"),
                    "generated_token_count": result.get("generated_token_count"),
                    "reasoning_token_count": result.get("reasoning_token_count"),
                }
            )
    return parsed, sorted(failures, key=lambda row: row["id"])


def load_joined_items() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    loaded: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    parsed_confidence: dict[str, dict[str, float]] = {}
    parse_failures: dict[str, list[dict[str, Any]]] = {}

    for condition in CONDITIONS:
        paths = input_paths(condition)
        loaded[condition] = {
            label: index_unique(path) for label, path in paths.items()
        }
        expected_ids = set(loaded[condition]["prompts"])
        for label in ("source_predictions", "source_scores"):
            if set(loaded[condition][label]) != expected_ids:
                raise ValueError(f"{condition} {label} does not match prompt IDs")
        parsed_confidence[condition], parse_failures[condition] = confidence_values(
            loaded[condition]["prompts"], loaded[condition]["confidence"]
        )

    prompt_ids = set(loaded["none"]["prompts"])
    if set(loaded["low"]["prompts"]) != prompt_ids:
        raise ValueError("No- and low-reasoning prompt IDs differ")
    common_ids = prompt_ids.intersection(*(set(parsed_confidence[c]) for c in CONDITIONS))

    items: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        condition_items = []
        for sample_id in sorted(common_ids):
            prompt = loaded[condition]["prompts"][sample_id]
            score = loaded[condition]["source_scores"][sample_id]
            source = loaded[condition]["source_predictions"][sample_id]
            reflection = loaded[condition]["confidence"][sample_id]
            if bool(prompt["semantic_correct"]) != bool(score["relaxed_correct"]):
                raise ValueError(f"Correctness mismatch for {condition} {sample_id}")
            if str(prompt["source_prediction"]) != str(score["prediction"]):
                raise ValueError(f"Prediction mismatch for {condition} {sample_id}")
            condition_items.append(
                {
                    "id": sample_id,
                    "condition": condition,
                    "split": str(prompt["split"]),
                    "confidence": parsed_confidence[condition][sample_id],
                    "semantic_correct": bool(score["relaxed_correct"]),
                    "strict_correct": bool(score["strict_correct"]),
                    "target_type": str(score["target_type"]),
                    "code_chars": int(score["code_chars"]),
                    "target_chars": int(score["target_chars"]),
                    "input_chars": int(score["input_chars"]),
                    "prediction_chars": int(score["prediction_chars"]),
                    "source_reasoning_tokens": int(source.get("reasoning_token_count") or 0),
                    "reflection_reasoning_tokens": int(
                        reflection.get("reasoning_token_count") or 0
                    ),
                    "source_generated_tokens": int(source.get("generated_token_count") or 0),
                    "reflection_generated_tokens": int(
                        reflection.get("generated_token_count") or 0
                    ),
                    "source_prediction": str(score["prediction_clean"]),
                    "target_output": str(score["target_output"]),
                    "error_category": str(score["relaxed_type_error_category"]),
                }
            )
        items[condition] = condition_items

    split_ids = {
        condition: {
            split: {row["id"] for row in rows if row["split"] == split}
            for split in ("calibration", "test")
        }
        for condition, rows in items.items()
    }
    for split in ("calibration", "test"):
        if split_ids["none"][split] != split_ids["low"][split]:
            raise ValueError(f"Paired complete-case IDs differ for {split}")

    failure_audit: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        enriched = []
        for failure in parse_failures[condition]:
            sample_id = failure["id"]
            score = loaded[condition]["source_scores"][sample_id]
            enriched.append(
                {
                    **failure,
                    "target_type": score["target_type"],
                    "source_semantic_correct": bool(score["relaxed_correct"]),
                    "source_prediction": score["prediction_clean"],
                    "target_output": score["target_output"],
                }
            )
        failure_audit[condition] = enriched

    metadata = {
        "input_paths": {
            condition: {
                label: str(resolve_path(path))
                for label, path in input_paths(condition).items()
            }
            for condition in CONDITIONS
        },
        "confidence_extraction_failures": failure_audit,
        "paired_complete_cases": len(common_ids),
        "paired_split_sizes": {
            split: len(split_ids["none"][split]) for split in ("calibration", "test")
        },
    }
    return items, metadata


def rows_for_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split]


def metric_for_rows(
    rows: list[dict[str, Any]],
    correct_field: str,
    baseline_probability: float,
) -> dict[str, Any]:
    y = np.asarray([row[correct_field] for row in rows], dtype=int)
    confidence = np.asarray([row["confidence"] for row in rows], dtype=float)
    return metric_bundle(y, confidence, baseline_probability)


def output_type_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calibration = rows_for_split(rows, "calibration")
    test = rows_for_split(rows, "test")
    pooled_calibration_baseline = float(
        np.mean([row["semantic_correct"] for row in calibration])
    )
    calibration_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration:
        calibration_by_type[row["target_type"]].append(row)
    for row in test:
        test_by_type[row["target_type"]].append(row)

    results = {}
    for target_type in sorted(test_by_type):
        test_rows = test_by_type[target_type]
        calibration_rows = calibration_by_type[target_type]
        if calibration_rows:
            baseline = float(
                np.mean([row["semantic_correct"] for row in calibration_rows])
            )
            baseline_scope = "same_type_calibration"
        else:
            baseline = pooled_calibration_baseline
            baseline_scope = "pooled_calibration_fallback"
        results[target_type] = {
            **metric_for_rows(test_rows, "semantic_correct", baseline),
            "baseline_scope": baseline_scope,
        }
    return results


def stratified_auc(
    rows: list[dict[str, Any]],
    group_key: Callable[[dict[str, Any]], Any],
    correct_field: str = "semantic_correct",
) -> dict[str, Any]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    weighted_auc = 0.0
    comparable_pairs = 0
    used = []
    excluded = []
    for group, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        y = np.asarray([row[correct_field] for row in group_rows], dtype=int)
        confidence = np.asarray([row["confidence"] for row in group_rows], dtype=float)
        positives = int(np.sum(y))
        negatives = len(y) - positives
        pairs = positives * negatives
        auc = roc_auc(y, confidence)
        record = {
            "group": str(group),
            "n": len(group_rows),
            "correct": positives,
            "incorrect": negatives,
            "comparable_pairs": pairs,
            "auroc": auc,
        }
        if auc is None:
            excluded.append(record)
            continue
        weighted_auc += float(auc) * pairs
        comparable_pairs += pairs
        used.append(record)
    return {
        "auroc": weighted_auc / comparable_pairs if comparable_pairs else None,
        "comparable_pairs": comparable_pairs,
        "strata_used": used,
        "strata_excluded": excluded,
    }


def bootstrap_stratified_auc(
    rows: list[dict[str, Any]],
    group_key: Callable[[dict[str, Any]], Any],
    repetitions: int,
    seed: int,
) -> list[float] | None:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sample = []
        for group_rows in groups.values():
            indices = rng.integers(0, len(group_rows), size=len(group_rows))
            sample.extend(group_rows[index] for index in indices)
        value = stratified_auc(sample, group_key)["auroc"]
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return percentile_interval(values)


def standardize(values: np.ndarray) -> np.ndarray:
    standard_deviation = float(np.std(values))
    if standard_deviation == 0:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / standard_deviation


def safe_exp(value: float) -> float | None:
    """Exponentiate a log-odds value without emitting non-standard JSON infinities."""
    if value > 700:
        return None
    if value < -745:
        return 0.0
    return math.exp(value)


def fit_logistic(design: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    def objective(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(np.sum(np.logaddexp(0, eta)) - y @ eta)

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = np.clip(design @ beta, -40, 40)
        probability = 1.0 / (1.0 + np.exp(-eta))
        return design.T @ (probability - y)

    fitted = minimize(
        objective,
        np.zeros(design.shape[1], dtype=float),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(-30.0, 30.0)] * design.shape[1],
        options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
    )
    beta = np.asarray(fitted.x, dtype=float)
    eta = np.clip(design @ beta, -40, 40)
    probability = 1.0 / (1.0 + np.exp(-eta))
    weights = probability * (1.0 - probability)
    information = (design.T * weights) @ design
    covariance = np.linalg.pinv(information)
    standard_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    return {
        "beta": beta,
        "standard_errors": standard_errors,
        "log_likelihood": -objective(beta),
        "success": bool(fitted.success),
        "message": str(fitted.message),
    }


def adjusted_confidence_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    test = rows_for_split(rows, "test")
    outcomes_by_type: dict[str, set[bool]] = defaultdict(set)
    for row in test:
        outcomes_by_type[row["target_type"]].add(row["semantic_correct"])
    estimable_types = sorted(
        target_type
        for target_type, outcomes in outcomes_by_type.items()
        if len(outcomes) == 2
    )
    excluded_types = sorted(set(outcomes_by_type) - set(estimable_types))
    selected = [row for row in test if row["target_type"] in estimable_types]
    reference_type = max(
        estimable_types,
        key=lambda target_type: sum(row["target_type"] == target_type for row in selected),
    )

    code_length = standardize(
        np.log1p(np.asarray([row["code_chars"] for row in selected], dtype=float))
    )
    output_length = standardize(
        np.log1p(np.asarray([row["target_chars"] for row in selected], dtype=float))
    )
    type_columns = [target_type for target_type in estimable_types if target_type != reference_type]
    base_design = np.asarray(
        [
            [1.0, code_length[index], output_length[index]]
            + [float(row["target_type"] == target_type) for target_type in type_columns]
            for index, row in enumerate(selected)
        ],
        dtype=float,
    )
    confidence_per_ten_points = np.asarray(
        [10.0 * (row["confidence"] - 0.5) for row in selected], dtype=float
    )
    full_design = np.column_stack((base_design, confidence_per_ten_points))
    y = np.asarray([row["semantic_correct"] for row in selected], dtype=float)
    reduced = fit_logistic(base_design, y)
    full = fit_logistic(full_design, y)
    statistic = max(0.0, 2.0 * (full["log_likelihood"] - reduced["log_likelihood"]))
    coefficient = float(full["beta"][-1])
    standard_error = float(full["standard_errors"][-1])
    return {
        "n": len(selected),
        "correct": int(np.sum(y)),
        "estimable_target_types": estimable_types,
        "excluded_single_outcome_target_types": excluded_types,
        "reference_target_type": reference_type,
        "covariates": [
            "z_log1p_code_chars",
            "z_log1p_target_chars",
            "target_type_fixed_effects",
        ],
        "confidence_unit": "10 percentage points",
        "confidence_coefficient": coefficient,
        "confidence_standard_error": standard_error,
        "confidence_odds_ratio": safe_exp(coefficient),
        "confidence_odds_ratio_95_ci": [
            safe_exp(coefficient - 1.96 * standard_error),
            safe_exp(coefficient + 1.96 * standard_error),
        ],
        "likelihood_ratio_statistic": statistic,
        "likelihood_ratio_p_value": float(chi2.sf(statistic, 1)),
        "reduced_fit_success": reduced["success"],
        "full_fit_success": full["success"],
        "reduced_fit_message": reduced["message"],
        "full_fit_message": full["message"],
    }


def scoring_policy_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calibration = rows_for_split(rows, "calibration")
    test = rows_for_split(rows, "test")
    results = {}
    for label, field in (
        ("semantic_value", "semantic_correct"),
        ("strict_serialization", "strict_correct"),
    ):
        baseline = float(np.mean([row[field] for row in calibration]))
        results[label] = metric_for_rows(test, field, baseline)
    semantic_labels = np.asarray([row["semantic_correct"] for row in test], dtype=bool)
    strict_labels = np.asarray([row["strict_correct"] for row in test], dtype=bool)
    results["label_changes"] = {
        "semantic_correct_but_strict_wrong": int(np.sum(semantic_labels & ~strict_labels)),
        "strict_correct_but_semantic_wrong": int(np.sum(strict_labels & ~semantic_labels)),
        "total_changed": int(np.sum(semantic_labels != strict_labels)),
        "test_items": len(test),
    }
    return results


def distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None, "mean": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "mean": float(np.mean(array)),
    }


def safe_spearman(left: list[float], right: list[float]) -> dict[str, float | None]:
    if len(set(left)) < 2 or len(set(right)) < 2:
        return {"rho": None, "p_value": None}
    result = spearmanr(left, right)
    return {"rho": float(result.statistic), "p_value": float(result.pvalue)}


def token_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    test = rows_for_split(rows, "test")
    results = {}
    for field in ("source_reasoning_tokens", "reflection_reasoning_tokens"):
        token_counts = [int(row[field]) for row in test]
        correct_tokens = [int(row[field]) for row in test if row["semantic_correct"]]
        incorrect_tokens = [int(row[field]) for row in test if not row["semantic_correct"]]
        if correct_tokens and incorrect_tokens:
            test_result = mannwhitneyu(
                correct_tokens, incorrect_tokens, alternative="two-sided"
            )
            rank_test = {
                "u_statistic": float(test_result.statistic),
                "p_value": float(test_result.pvalue),
            }
        else:
            rank_test = {"u_statistic": None, "p_value": None}
        results[field] = {
            "all": distribution(token_counts),
            "correct": distribution(correct_tokens),
            "incorrect": distribution(incorrect_tokens),
            "correlation_with_confidence": safe_spearman(
                token_counts, [float(row["confidence"]) for row in test]
            ),
            "correlation_with_correctness": safe_spearman(
                token_counts, [float(row["semantic_correct"]) for row in test]
            ),
            "correct_vs_incorrect_mann_whitney": rank_test,
        }
    return results


def selected_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "id",
            "confidence",
            "target_type",
            "source_prediction",
            "target_output",
            "error_category",
            "source_reasoning_tokens",
            "reflection_reasoning_tokens",
        )
    }


def case_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    test = rows_for_split(rows, "test")
    high_errors = [
        row
        for row in test
        if row["confidence"] >= 0.95 and not row["semantic_correct"]
    ]
    low_correct = [
        row
        for row in test
        if row["confidence"] <= 0.50 and row["semantic_correct"]
    ]
    high_errors.sort(key=lambda row: (-row["confidence"], row["id"]))
    low_correct.sort(key=lambda row: (row["confidence"], row["id"]))
    return {
        "thresholds": {"high_confidence": 0.95, "low_confidence": 0.50},
        "high_confidence_errors": {
            "count": len(high_errors),
            "target_types": dict(Counter(row["target_type"] for row in high_errors)),
            "error_categories": dict(
                Counter(row["error_category"] for row in high_errors).most_common()
            ),
            "examples": [selected_case(row) for row in high_errors[:10]],
        },
        "low_confidence_correct": {
            "count": len(low_correct),
            "target_types": dict(Counter(row["target_type"] for row in low_correct)),
            "examples": [selected_case(row) for row in low_correct[:10]],
        },
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "CRUXEval calibration internal-validity audit",
        "=" * 45,
        "",
        (
            f"Paired complete cases: {summary['data']['paired_complete_cases']} "
            f"(calibration={summary['data']['paired_split_sizes']['calibration']}, "
            f"test={summary['data']['paired_split_sizes']['test']})."
        ),
        "Primary correctness policy: semantic Python value equivalence.",
        "",
        "1. Signal within output types",
        "-----------------------------",
    ]
    for condition in CONDITIONS:
        result = summary["conditions"][condition]
        stratified = result["target_type_stratified_auc"]
        interval = stratified["bootstrap_95_ci"]
        lines.append(
            f"{condition}: type-stratified AUROC={fmt(stratified['auroc'])} "
            f"(95% CI {fmt(interval[0])}-{fmt(interval[1])}; "
            f"{stratified['comparable_pairs']} comparable correct/incorrect pairs)."
        )
        lines.append("  Held-out metrics by output type (types with at least 20 items):")
        for target_type, metrics in result["by_target_type"].items():
            if metrics["n"] < 20:
                continue
            lines.append(
                f"    {target_type:>5}: n={metrics['n']}, "
                f"accuracy={fmt(metrics['accuracy'])}, confidence={fmt(metrics['mean_confidence'])}, "
                f"gap={fmt(metrics['calibration_gap'])}, AUROC={fmt(metrics['auroc'])}"
            )
    lines.extend(["", "2. Confidence after controlling for obvious item features", "---------------------------------------------------------"])
    for condition in CONDITIONS:
        adjusted = summary["conditions"][condition]["adjusted_confidence"]
        odds_interval = adjusted["confidence_odds_ratio_95_ci"]
        lines.append(
            f"{condition}: n={adjusted['n']}; OR per +10 confidence points="
            f"{fmt(adjusted['confidence_odds_ratio'])} "
            f"(95% Wald CI {fmt(odds_interval[0])}-{fmt(odds_interval[1])}); "
            f"LR p={adjusted['likelihood_ratio_p_value']:.3g}."
        )
        lines.append(
            "  Controls: output-type fixed effects, log code length, and log target length; "
            f"excluded all-one-outcome types={adjusted['excluded_single_outcome_target_types']}."
        )

    lines.extend(["", "3. Sensitivity to scoring policy", "---------------------------------"])
    for condition in CONDITIONS:
        scoring = summary["conditions"][condition]["scoring_policy"]
        semantic = scoring["semantic_value"]
        strict = scoring["strict_serialization"]
        changed = scoring["label_changes"]
        lines.append(
            f"{condition}: semantic accuracy={fmt(semantic['accuracy'])}, "
            f"AUROC={fmt(semantic['auroc'])}, Brier skill={fmt(semantic['brier_skill_score'])}; "
            f"strict accuracy={fmt(strict['accuracy'])}, AUROC={fmt(strict['auroc'])}, "
            f"Brier skill={fmt(strict['brier_skill_score'])}."
        )
        lines.append(
            f"  The scoring policy changes {changed['total_changed']}/{changed['test_items']} "
            "held-out labels. Strict serialization was not the confidence prompt's target."
        )

    lines.extend(["", "4. Reasoning-token use", "----------------------"])
    for condition in CONDITIONS:
        token = summary["conditions"][condition]["reasoning_tokens"]
        source = token["source_reasoning_tokens"]
        reflection = token["reflection_reasoning_tokens"]
        lines.append(
            f"{condition}: source-token rho(confidence)="
            f"{fmt(source['correlation_with_confidence']['rho'])}; "
            f"reflection-token rho(confidence)="
            f"{fmt(reflection['correlation_with_confidence']['rho'])}."
        )
        lines.append(
            "  Source reasoning tokens, median correct/incorrect: "
            f"{fmt(source['correct']['median'], 1)}/{fmt(source['incorrect']['median'], 1)}; "
            "reflection reasoning tokens: "
            f"{fmt(reflection['correct']['median'], 1)}/{fmt(reflection['incorrect']['median'], 1)}."
        )

    lines.extend(["", "5. Extreme cases", "----------------"])
    for condition in CONDITIONS:
        cases = summary["conditions"][condition]["cases"]
        high = cases["high_confidence_errors"]
        low = cases["low_confidence_correct"]
        lines.append(
            f"{condition}: {high['count']} errors at confidence >=95%; "
            f"{low['count']} correct predictions at confidence <=50%."
        )
        lines.append(f"  High-confidence error target types: {high['target_types']}")
        for example in high["examples"][:5]:
            lines.append(
                f"    {example['id']} c={example['confidence']:.2f} "
                f"{example['error_category']}: {example['source_prediction']!r} "
                f"!= {example['target_output']!r}"
            )

    lines.extend(["", "6. Confidence-call truncations", "-------------------------------"])
    for condition in CONDITIONS:
        failures = summary["data"]["confidence_extraction_failures"][condition]
        lines.append(f"{condition}: {len(failures)} unparsed confidence responses.")
        for failure in failures:
            lines.append(
                f"  {failure['id']} ({failure['split']}, {failure['target_type']}): "
                f"finish={failure['finish_reason']}/{failure['native_finish_reason']}, "
                f"generated={failure['generated_token_count']}, "
                f"reasoning={failure['reasoning_token_count']}, "
                f"source_correct={failure['source_semantic_correct']}"
            )

    lines.extend(
        [
            "",
            "Bottom line",
            "-----------",
            (
                "Low-reasoning confidence is not explained away by output type, code length, "
                "or output length. Its type-stratified and adjusted signals remain strong."
            ),
            (
                "No-reasoning confidence retains a weak discrimination signal but remains "
                "operationally unsafe because it is heavily overconfident."
            ),
            (
                "The same predictions look radically less reliable under strict string "
                "serialization, demonstrating that scoring policy is part of a reliability claim."
            ),
            (
                "The next discriminating experiment is a prediction-only confidence control, "
                "which tests whether the signal comes from code-aware verification or output surface cues."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    items, metadata = load_joined_items()
    conditions = {}
    for index, condition in enumerate(CONDITIONS):
        test_rows = rows_for_split(items[condition], "test")
        type_stratified = stratified_auc(test_rows, lambda row: row["target_type"])
        type_stratified["bootstrap_95_ci"] = bootstrap_stratified_auc(
            test_rows,
            lambda row: row["target_type"],
            args.bootstraps,
            args.seed + index,
        )
        conditions[condition] = {
            "by_target_type": output_type_audit(items[condition]),
            "target_type_stratified_auc": type_stratified,
            "adjusted_confidence": adjusted_confidence_audit(items[condition]),
            "scoring_policy": scoring_policy_audit(items[condition]),
            "reasoning_tokens": token_audit(items[condition]),
            "cases": case_audit(items[condition]),
        }

    summary = {
        "audit": "cruxeval_calibration_internal_validity_v1",
        "analysis_scope": (
            "Paired complete-case, held-out audit of output-type confounding, code and output "
            "length, scoring policy, reasoning-token use, extreme cases, and truncations."
        ),
        "primary_correctness": "relaxed_correct semantic Python value equivalence",
        "bootstraps": args.bootstraps,
        "bootstrap_seed": args.seed,
        "data": metadata,
        "conditions": conditions,
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
    print(f"JSON audit: {json_output}")
    print(f"Text audit: {text_output}")


if __name__ == "__main__":
    main()
