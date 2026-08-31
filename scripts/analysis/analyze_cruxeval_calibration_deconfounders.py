"""Test whether CRUXEval calibration findings reduce to prior or composition shift."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import chi2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from analyze_cruxeval_calibration_pilot import (  # noqa: E402
    metric_bundle,
    percentile_interval,
    roc_auc,
)
from audit_cruxeval_calibration_internal_validity import (  # noqa: E402
    CONDITIONS,
    load_joined_items,
    stratified_auc,
)


DEFAULT_BOOTSTRAPS = 2000
DEFAULT_SEED = 20260818
CLIP = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "research_design/2026-08-18_cruxeval_calibration_deconfounding_analysis.json"
        ),
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=Path(
            "research_design/2026-08-18_cruxeval_calibration_deconfounding_results.txt"
        ),
    )
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def split_rows(
    items: dict[str, list[dict[str, Any]]], condition: str, split: str
) -> list[dict[str, Any]]:
    return sorted(
        (row for row in items[condition] if row["split"] == split),
        key=lambda row: row["id"],
    )


def arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([row["semantic_correct"] for row in rows], dtype=int),
        np.asarray([row["confidence"] for row in rows], dtype=float),
    )


def confidence_logit(probability: np.ndarray) -> np.ndarray:
    return logit(np.clip(probability, CLIP, 1.0 - CLIP))


def negative_log_likelihood(y: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return -float(np.sum(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))


def mean_log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    return negative_log_likelihood(y, probability) / len(y)


def fit_calibration_model(
    probability: np.ndarray,
    y: np.ndarray,
    fixed_slope: float | None = None,
) -> dict[str, Any]:
    x = confidence_logit(probability)

    if fixed_slope is not None:
        slope = float(fixed_slope)

        def objective(intercept: float) -> float:
            eta = slope * x + intercept
            return float(np.sum(np.logaddexp(0, eta)) - y @ eta)

        fitted = minimize_scalar(
            objective, bounds=(-20.0, 20.0), method="bounded"
        )
        return {
            "intercept": float(fitted.x),
            "slope": slope,
            "negative_log_likelihood": float(fitted.fun),
            "success": bool(fitted.success),
            "message": str(fitted.message),
            "slope_fixed": True,
        }

    def objective(theta: np.ndarray) -> float:
        eta = theta[0] + theta[1] * x
        return float(np.sum(np.logaddexp(0, eta)) - y @ eta)

    def gradient(theta: np.ndarray) -> np.ndarray:
        eta = np.clip(theta[0] + theta[1] * x, -40, 40)
        residual = expit(eta) - y
        return np.asarray([np.sum(residual), residual @ x], dtype=float)

    fitted = minimize(
        objective,
        np.asarray([0.0, 1.0]),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0), (-20.0, 20.0)],
        options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
    )
    return {
        "intercept": float(fitted.x[0]),
        "slope": float(fitted.x[1]),
        "negative_log_likelihood": float(fitted.fun),
        "success": bool(fitted.success),
        "message": str(fitted.message),
        "slope_fixed": False,
    }


def apply_calibration(
    probability: np.ndarray, model: dict[str, Any]
) -> np.ndarray:
    return expit(
        float(model["intercept"])
        + float(model["slope"]) * confidence_logit(probability)
    )


def evaluated_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    calibration_prevalence: float,
) -> dict[str, Any]:
    return {
        **metric_bundle(y, probability, calibration_prevalence),
        "log_loss": mean_log_loss(y, probability),
    }


def bootstrap_transfer_pipeline(
    source_calibration_y: np.ndarray,
    source_calibration_probability: np.ndarray,
    target_calibration_y: np.ndarray,
    target_calibration_probability: np.ndarray,
    target_test_y: np.ndarray,
    target_test_probability: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    brier_differences = []
    log_loss_differences = []
    slope_differences = []
    for _ in range(repetitions):
        source_indices = rng.integers(
            0, len(source_calibration_y), size=len(source_calibration_y)
        )
        target_calibration_indices = rng.integers(
            0, len(target_calibration_y), size=len(target_calibration_y)
        )
        target_test_indices = rng.integers(
            0, len(target_test_y), size=len(target_test_y)
        )
        source_model = fit_calibration_model(
            source_calibration_probability[source_indices],
            source_calibration_y[source_indices],
        )
        target_model = fit_calibration_model(
            target_calibration_probability[target_calibration_indices],
            target_calibration_y[target_calibration_indices],
        )
        source_slope_target_intercept = fit_calibration_model(
            target_calibration_probability[target_calibration_indices],
            target_calibration_y[target_calibration_indices],
            fixed_slope=float(source_model["slope"]),
        )
        sampled_y = target_test_y[target_test_indices]
        source_probability = apply_calibration(
            target_test_probability[target_test_indices],
            source_slope_target_intercept,
        )
        target_probability = apply_calibration(
            target_test_probability[target_test_indices], target_model
        )
        brier_differences.append(
            float(
                np.mean((source_probability - sampled_y) ** 2)
                - np.mean((target_probability - sampled_y) ** 2)
            )
        )
        log_loss_differences.append(
            mean_log_loss(sampled_y, source_probability)
            - mean_log_loss(sampled_y, target_probability)
        )
        slope_differences.append(
            float(target_model["slope"]) - float(source_model["slope"])
        )
    return {
        "definition": (
            "source-slope plus target-intercept loss minus target-refit loss; "
            "positive values favor refitting the slope"
        ),
        "resampling": (
            "Independently resample source calibration, target calibration, and target test "
            "items, then refit every calibration model in each bootstrap replicate."
        ),
        "brier_difference_95_ci": percentile_interval(brier_differences),
        "log_loss_difference_95_ci": percentile_interval(log_loss_differences),
        "target_minus_source_slope_95_ci": percentile_interval(slope_differences),
    }


def transfer_decomposition(
    items: dict[str, list[dict[str, Any]]],
    source: str,
    target: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    source_calibration_rows = split_rows(items, source, "calibration")
    target_calibration_rows = split_rows(items, target, "calibration")
    target_test_rows = split_rows(items, target, "test")
    source_y, source_probability = arrays(source_calibration_rows)
    target_calibration_y, target_calibration_probability = arrays(
        target_calibration_rows
    )
    target_test_y, target_test_probability = arrays(target_test_rows)
    target_prevalence = float(np.mean(target_calibration_y))

    source_model = fit_calibration_model(source_probability, source_y)
    target_model = fit_calibration_model(
        target_calibration_probability, target_calibration_y
    )
    source_slope_target_intercept = fit_calibration_model(
        target_calibration_probability,
        target_calibration_y,
        fixed_slope=float(source_model["slope"]),
    )

    uncorrected_probability = apply_calibration(target_test_probability, source_model)
    intercept_updated_probability = apply_calibration(
        target_test_probability, source_slope_target_intercept
    )
    target_refit_probability = apply_calibration(target_test_probability, target_model)
    raw_probability = target_test_probability

    likelihood_ratio = max(
        0.0,
        2.0
        * (
            float(source_slope_target_intercept["negative_log_likelihood"])
            - float(target_model["negative_log_likelihood"])
        ),
    )
    bootstrap = bootstrap_transfer_pipeline(
        source_y,
        source_probability,
        target_calibration_y,
        target_calibration_probability,
        target_test_y,
        target_test_probability,
        repetitions,
        seed,
    )
    intercept_metrics = evaluated_metrics(
        target_test_y, intercept_updated_probability, target_prevalence
    )
    target_metrics = evaluated_metrics(
        target_test_y, target_refit_probability, target_prevalence
    )
    bootstrap.update(
        {
            "brier_difference_point": (
                float(intercept_metrics["brier_score"])
                - float(target_metrics["brier_score"])
            ),
            "log_loss_difference_point": (
                float(intercept_metrics["log_loss"])
                - float(target_metrics["log_loss"])
            ),
        }
    )
    return {
        "source_condition": source,
        "target_condition": target,
        "source_calibration_prevalence": float(np.mean(source_y)),
        "target_calibration_prevalence": target_prevalence,
        "source_full_model": source_model,
        "source_slope_target_intercept_model": source_slope_target_intercept,
        "target_full_model": target_model,
        "target_calibration_slope_likelihood_ratio": {
            "statistic": likelihood_ratio,
            "p_value": float(chi2.sf(likelihood_ratio, 1)),
            "null": "target calibration slope equals source calibration slope",
        },
        "held_out_target": {
            "raw": evaluated_metrics(
                target_test_y, raw_probability, target_prevalence
            ),
            "source_full_uncorrected": evaluated_metrics(
                target_test_y, uncorrected_probability, target_prevalence
            ),
            "source_slope_target_intercept": intercept_metrics,
            "target_full_refit": target_metrics,
        },
        "held_out_residual_slope_penalty": bootstrap,
    }


def calibration_decomposition(
    items: dict[str, list[dict[str, Any]]], condition: str
) -> dict[str, Any]:
    calibration_rows = split_rows(items, condition, "calibration")
    test_rows = split_rows(items, condition, "test")
    calibration_y, calibration_probability = arrays(calibration_rows)
    test_y, test_probability = arrays(test_rows)
    prevalence = float(np.mean(calibration_y))
    calibration_in_the_large = fit_calibration_model(
        calibration_probability, calibration_y, fixed_slope=1.0
    )
    full_model = fit_calibration_model(calibration_probability, calibration_y)
    return {
        "calibration_prevalence": prevalence,
        "test_prevalence": float(np.mean(test_y)),
        "raw_test": evaluated_metrics(test_y, test_probability, prevalence),
        "calibration_in_the_large_model": calibration_in_the_large,
        "full_calibration_model": full_model,
        "full_model_test": evaluated_metrics(
            test_y, apply_calibration(test_probability, full_model), prevalence
        ),
    }


def paired_auc_bootstrap(
    items: dict[str, list[dict[str, Any]]], repetitions: int, seed: int
) -> dict[str, Any]:
    by_id = {
        condition: {
            row["id"]: row for row in split_rows(items, condition, "test")
        }
        for condition in CONDITIONS
    }
    sample_ids = sorted(set(by_id["none"]) & set(by_id["low"]))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        sampled_ids = [sample_ids[index] for index in rng.integers(0, len(sample_ids), len(sample_ids))]
        aucs = {}
        for condition in CONDITIONS:
            y = np.asarray(
                [by_id[condition][sample_id]["semantic_correct"] for sample_id in sampled_ids],
                dtype=int,
            )
            probability = np.asarray(
                [by_id[condition][sample_id]["confidence"] for sample_id in sampled_ids],
                dtype=float,
            )
            aucs[condition] = roc_auc(y, probability)
        if aucs["none"] is not None and aucs["low"] is not None:
            differences.append(float(aucs["low"] - aucs["none"]))

    observed = {}
    for condition in CONDITIONS:
        rows = [by_id[condition][sample_id] for sample_id in sample_ids]
        y, probability = arrays(rows)
        observed[condition] = roc_auc(y, probability)
    return {
        "n_paired_items": len(sample_ids),
        "none_auroc": observed["none"],
        "low_auroc": observed["low"],
        "low_minus_none_point": float(observed["low"] - observed["none"]),
        "low_minus_none_95_ci": percentile_interval(differences),
        "note": "AUROC is prevalence-invariant; this bootstrap preserves paired item sampling.",
    }


def shared_type_counts(
    test_rows: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    target_types = sorted(
        set(row["target_type"] for condition in CONDITIONS for row in test_rows[condition])
    )
    return {
        target_type: min(
            sum(
                row["target_type"] == target_type
                and row["semantic_correct"] == outcome
                for row in test_rows[condition]
            )
            for condition in CONDITIONS
            for outcome in (False, True)
        )
        for target_type in target_types
    }


def matched_discrimination_bootstrap(
    items: dict[str, list[dict[str, Any]]], repetitions: int, seed: int
) -> dict[str, Any]:
    test_rows = {
        condition: split_rows(items, condition, "test") for condition in CONDITIONS
    }
    counts = shared_type_counts(test_rows)
    usable_counts = {key: value for key, value in counts.items() if value > 0}
    sample_size = 2 * sum(usable_counts.values())
    rng = np.random.default_rng(seed)
    collected = {"none": [], "low": [], "difference": []}
    for _ in range(repetitions):
        aucs = {}
        for condition in CONDITIONS:
            sampled_rows = []
            for target_type, count in usable_counts.items():
                for outcome in (False, True):
                    eligible = [
                        row
                        for row in test_rows[condition]
                        if row["target_type"] == target_type
                        and row["semantic_correct"] == outcome
                    ]
                    indices = rng.integers(0, len(eligible), size=count)
                    sampled_rows.extend(eligible[index] for index in indices)
            y, probability = arrays(sampled_rows)
            aucs[condition] = roc_auc(y, probability)
            if aucs[condition] is None:
                raise AssertionError("Matched sample must contain both outcomes")
            collected[condition].append(float(aucs[condition]))
        collected["difference"].append(float(aucs["low"] - aucs["none"]))

    return {
        "matching_policy": (
            "For each output type, use the minimum correct/incorrect count shared by both "
            "reasoning conditions; sample that count per class and condition with replacement."
        ),
        "per_condition_sample_size": sample_size,
        "correctness_prevalence": 0.5,
        "per_class_output_type_counts": usable_counts,
        "none_auroc_median": float(np.median(collected["none"])),
        "none_auroc_95_interval": percentile_interval(collected["none"]),
        "low_auroc_median": float(np.median(collected["low"])),
        "low_auroc_95_interval": percentile_interval(collected["low"]),
        "low_minus_none_median": float(np.median(collected["difference"])),
        "low_minus_none_95_interval": percentile_interval(collected["difference"]),
        "limitations": (
            "The common support is dominated by string errors and remains exploratory; "
            "a second benchmark is needed for confirmation."
        ),
    }


def wilson_interval(correct: int, total: int, z: float = 1.96) -> list[float]:
    proportion = correct / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return [centre - radius, centre + radius]


def fixed_coverage_selective_risk(
    items: dict[str, list[dict[str, Any]]], condition: str
) -> dict[str, Any]:
    rows = split_rows(items, condition, "test")
    y, probability = arrays(rows)
    full_accuracy = float(np.mean(y))
    full_risk = 1.0 - full_accuracy
    ordered_probability = np.sort(probability)[::-1]
    results = []
    for requested_coverage in (0.70, 0.80, 0.90):
        requested_count = int(math.ceil(len(rows) * requested_coverage))
        threshold = float(ordered_probability[requested_count - 1])
        accepted = probability >= threshold
        accepted_count = int(np.sum(accepted))
        correct = int(np.sum(y[accepted]))
        accuracy = correct / accepted_count
        risk = 1.0 - accuracy
        results.append(
            {
                "requested_coverage": requested_coverage,
                "actual_coverage": accepted_count / len(rows),
                "accepted": accepted_count,
                "minimum_confidence": threshold,
                "accuracy": accuracy,
                "accuracy_wilson_95_ci": wilson_interval(correct, accepted_count),
                "risk": risk,
                "relative_error_reduction_vs_answer_all": (
                    (full_risk - risk) / full_risk if full_risk else None
                ),
            }
        )
    return {
        "n": len(rows),
        "answer_all_accuracy": full_accuracy,
        "answer_all_risk": full_risk,
        "tie_policy": "Do not split equal-confidence groups; actual coverage may exceed requested.",
        "fixed_coverage": results,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "CRUXEval calibration deconfounding audit",
        "=" * 41,
        "",
        "Exploratory reanalysis of the already-inspected pilot test split.",
        "No new model calls were made.",
        "",
        "1. Calibration intercept and slope",
        "----------------------------------",
    ]
    for condition in CONDITIONS:
        decomposition = summary["calibration_decomposition"][condition]
        full = decomposition["full_calibration_model"]
        lines.append(
            f"{condition}: calibration prevalence={fmt(decomposition['calibration_prevalence'])}, "
            f"raw test gap={fmt(decomposition['raw_test']['calibration_gap'])}, "
            f"calibration-in-the-large intercept={fmt(decomposition['calibration_in_the_large_model']['intercept'])}, "
            f"full slope={fmt(full['slope'])}, full intercept={fmt(full['intercept'])}."
        )

    lines.extend(["", "2. Does prior/intercept correction restore transfer?", "---------------------------------------------------"])
    for key in ("none->low", "low->none"):
        transfer = summary["transfer_decomposition"][key]
        held_out = transfer["held_out_target"]
        penalty = transfer["held_out_residual_slope_penalty"]
        interval = penalty["brier_difference_95_ci"]
        lines.append(
            f"{key}: uncorrected gap={fmt(held_out['source_full_uncorrected']['calibration_gap'])}, "
            f"Brier={fmt(held_out['source_full_uncorrected']['brier_score'])}; "
            f"after target-intercept update gap={fmt(held_out['source_slope_target_intercept']['calibration_gap'])}, "
            f"Brier={fmt(held_out['source_slope_target_intercept']['brier_score'])}; "
            f"target full-refit Brier={fmt(held_out['target_full_refit']['brier_score'])}."
        )
        lines.append(
            f"  Residual Brier penalty from retaining the source slope: "
            f"{fmt(penalty['brier_difference_point'])} "
            f"(95% CI {fmt(interval[0])}-{fmt(interval[1])}); "
            f"calibration-slope LR p={transfer['target_calibration_slope_likelihood_ratio']['p_value']:.3g}."
        )

    paired = summary["discrimination"]["paired_all_items"]
    matched = summary["discrimination"]["base_rate_and_type_matched"]
    paired_interval = paired["low_minus_none_95_ci"]
    matched_interval = matched["low_minus_none_95_interval"]
    lines.extend(
        [
            "",
            "3. Does the reasoning discrimination gain survive matching?",
            "-----------------------------------------------------------",
            (
                f"All paired held-out items: AUROC none={fmt(paired['none_auroc'])}, "
                f"low={fmt(paired['low_auroc'])}, delta={fmt(paired['low_minus_none_point'])} "
                f"(95% CI {fmt(paired_interval[0])}-{fmt(paired_interval[1])})."
            ),
            (
                f"Balanced/type-matched resamples: n={matched['per_condition_sample_size']} per condition, "
                f"50% correct; median AUROC none={fmt(matched['none_auroc_median'])}, "
                f"low={fmt(matched['low_auroc_median'])}, delta={fmt(matched['low_minus_none_median'])} "
                f"(95% interval {fmt(matched_interval[0])}-{fmt(matched_interval[1])})."
            ),
            f"Matched per-class type counts: {matched['per_class_output_type_counts']}",
            "",
            "4. Selective risk at fixed requested coverage",
            "---------------------------------------------",
        ]
    )
    for condition in CONDITIONS:
        selective = summary["selective_risk"][condition]
        lines.append(
            f"{condition}: answer-all accuracy={fmt(selective['answer_all_accuracy'])}, "
            f"risk={fmt(selective['answer_all_risk'])}."
        )
        for row in selective["fixed_coverage"]:
            lines.append(
                f"  requested {row['requested_coverage']:.0%}: actual={row['actual_coverage']:.1%}, "
                f"accuracy={row['accuracy']:.1%}, risk={row['risk']:.1%}, "
                f"relative error reduction={row['relative_error_reduction_vs_answer_all']:.1%}."
            )

    lines.extend(
        [
            "",
            "Conclusion",
            "----------",
            (
                "Prior shift explains most of the transferred mean-confidence error, but not "
                "the full transfer failure: retaining the source calibration slope leaves a "
                "positive held-out Brier and log-loss penalty in both directions."
            ),
            (
                "The low-reasoning discrimination advantage also survives an exactly balanced, "
                "output-type-matched comparison, so it is not arithmetic from the 48.9% versus "
                "88.9% correctness rates."
            ),
            (
                "Selective prediction is materially useful in the low-reasoning pilot at fixed "
                "coverage, but these results remain exploratory and require replication on a fresh "
                "benchmark/model because the existing test split has informed the analysis."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    items, metadata = load_joined_items()
    decomposition = {
        condition: calibration_decomposition(items, condition)
        for condition in CONDITIONS
    }
    transfers = {
        "none->low": transfer_decomposition(
            items, "none", "low", args.bootstraps, args.seed
        ),
        "low->none": transfer_decomposition(
            items, "low", "none", args.bootstraps, args.seed + 1
        ),
    }
    summary = {
        "analysis": "cruxeval_calibration_deconfounding_v1",
        "status": "exploratory_existing_test_reanalysis",
        "bootstraps": args.bootstraps,
        "bootstrap_seed": args.seed,
        "data": {
            "paired_complete_cases": metadata["paired_complete_cases"],
            "paired_split_sizes": metadata["paired_split_sizes"],
        },
        "calibration_decomposition": decomposition,
        "transfer_decomposition": transfers,
        "discrimination": {
            "paired_all_items": paired_auc_bootstrap(
                items, args.bootstraps, args.seed + 2
            ),
            "base_rate_and_type_matched": matched_discrimination_bootstrap(
                items, args.bootstraps, args.seed + 3
            ),
            "type_stratified_point": {
                condition: stratified_auc(
                    split_rows(items, condition, "test"),
                    lambda row: row["target_type"],
                )
                for condition in CONDITIONS
            },
        },
        "selective_risk": {
            condition: fixed_coverage_selective_risk(items, condition)
            for condition in CONDITIONS
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
