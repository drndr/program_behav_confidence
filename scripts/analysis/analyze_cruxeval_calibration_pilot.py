"""Analyze reflective confidence, calibration, and reasoning-setting transfer."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("none", "low")
DEFAULT_BOOTSTRAPS = 2000
DEFAULT_SEED = 20260817
CONFIDENCE_PATTERN = re.compile(
    r'["\']?confidence["\']?\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))',
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--none-prompts",
        type=Path,
        default=Path("data/prompts/cruxeval/cruxeval_reflective_confidence_none_v1.jsonl"),
    )
    parser.add_argument(
        "--low-prompts",
        type=Path,
        default=Path("data/prompts/cruxeval/cruxeval_reflective_confidence_low_v1.jsonl"),
    )
    parser.add_argument(
        "--none-confidence",
        type=Path,
        default=Path(
            "results/cruxeval_reflective_confidence_gpt_5_4_nano_none_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--low-confidence",
        type=Path,
        default=Path(
            "results/cruxeval_reflective_confidence_gpt_5_4_nano_low_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research_design/2026-08-17_cruxeval_calibration_pilot_manifest.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("research_design/2026-08-17_cruxeval_calibration_pilot_analysis.json"),
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=Path("research_design/2026-08-17_cruxeval_calibration_pilot_results.txt"),
    )
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--minimum-threshold-items", type=int, default=40)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = resolve_path(path)
    rows = []
    with resolved.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSON at {resolved}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {resolved}:{line_number}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No records found in {resolved}")
    return rows


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id:
            raise ValueError(f"{label} contains a row without an ID")
        if sample_id in indexed:
            raise ValueError(f"{label} contains duplicate ID {sample_id}")
        indexed[sample_id] = row
    return indexed


def parse_confidence(response: str) -> float:
    text = response.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    value: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        value = parsed.get("confidence")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        match = CONFIDENCE_PATTERN.search(text)
        if match:
            value = float(match.group(1))
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Could not parse confidence from {response!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or numeric > 100:
        raise ValueError(f"Confidence must be between 0 and 100, found {numeric}")
    return numeric / 100.0


def roc_auc(y: np.ndarray, probability: np.ndarray) -> float | None:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(probability, method="average")
    positive_rank_sum = float(np.sum(ranks[y == 1]))
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def equal_mass_ece(y: np.ndarray, probability: np.ndarray, bins: int = 5) -> float:
    order = np.argsort(probability, kind="stable")
    total = len(order)
    error = 0.0
    for indices in np.array_split(order, min(bins, total)):
        if len(indices) == 0:
            continue
        error += len(indices) / total * abs(
            float(np.mean(probability[indices])) - float(np.mean(y[indices]))
        )
    return error


def brier_score(y: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((probability - y) ** 2))


def metric_bundle(
    y: np.ndarray,
    probability: np.ndarray,
    baseline_probability: float,
) -> dict[str, float | int | None]:
    brier = brier_score(y, probability)
    baseline_vector = np.full(len(y), baseline_probability, dtype=float)
    baseline_brier = brier_score(y, baseline_vector)
    skill = None if baseline_brier == 0 else 1.0 - brier / baseline_brier
    return {
        "n": int(len(y)),
        "correct": int(np.sum(y)),
        "accuracy": float(np.mean(y)),
        "mean_confidence": float(np.mean(probability)),
        "calibration_gap": float(np.mean(probability) - np.mean(y)),
        "brier_score": brier,
        "baseline_probability": float(baseline_probability),
        "baseline_brier_score": baseline_brier,
        "brier_skill_score": skill,
        "ece_equal_mass_5": equal_mass_ece(y, probability, bins=5),
        "auroc": roc_auc(y, probability),
    }


def fit_platt(probability: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(probability, 1e-4, 1 - 1e-4)
    x = logit(clipped)
    prevalence = float(np.mean(y))
    if prevalence in {0.0, 1.0}:
        return {
            "slope": 0.0,
            "intercept": float(logit(np.clip(prevalence, 1e-6, 1 - 1e-6))),
            "success": True,
            "message": "constant-label calibration set",
        }

    def objective(theta: np.ndarray) -> float:
        predicted = expit(theta[0] * x + theta[1])
        predicted = np.clip(predicted, 1e-12, 1 - 1e-12)
        negative_log_likelihood = -float(
            np.sum(y * np.log(predicted) + (1 - y) * np.log(1 - predicted))
        )
        return negative_log_likelihood + 1e-5 * float(np.sum(theta**2))

    initial = np.array([1.0, 0.0], dtype=float)
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0), (-20.0, 20.0)],
    )
    if not fitted.success:
        fallback = np.array([0.0, logit(prevalence)], dtype=float)
        if objective(fallback) < objective(fitted.x):
            fitted.x = fallback
    return {
        "slope": float(fitted.x[0]),
        "intercept": float(fitted.x[1]),
        "success": bool(fitted.success),
        "message": str(fitted.message),
    }


def apply_platt(probability: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    x = logit(np.clip(probability, 1e-4, 1 - 1e-4))
    return expit(float(model["slope"]) * x + float(model["intercept"]))


def risk_coverage(y: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    ordered_probability = np.sort(probability)[::-1]
    rows = []
    for requested_coverage in (0.25, 0.50, 0.75, 1.0):
        requested_count = max(1, int(math.ceil(len(y) * requested_coverage)))
        threshold = float(ordered_probability[requested_count - 1])
        accepted_mask = probability >= threshold
        accepted = int(np.sum(accepted_mask))
        accuracy = float(np.mean(y[accepted_mask]))
        rows.append(
            {
                "requested_coverage": requested_coverage,
                "accepted": accepted,
                "actual_coverage": accepted / len(y),
                "accuracy": accuracy,
                "risk": 1.0 - accuracy,
                "minimum_confidence": threshold,
            }
        )
    return rows


def select_accuracy_threshold(
    y: np.ndarray,
    probability: np.ndarray,
    target_accuracy: float,
    minimum_items: int,
) -> dict[str, Any] | None:
    candidates = []
    for threshold in np.unique(probability):
        accepted = probability >= threshold
        count = int(np.sum(accepted))
        if count < minimum_items:
            continue
        accuracy = float(np.mean(y[accepted]))
        if accuracy >= target_accuracy:
            candidates.append((count, float(threshold), accuracy))
    if not candidates:
        return None
    count, threshold, accuracy = max(candidates, key=lambda row: (row[0], row[1]))
    return {
        "threshold": threshold,
        "accepted": count,
        "coverage": count / len(y),
        "accuracy": accuracy,
        "target_accuracy": target_accuracy,
        "minimum_items": minimum_items,
    }


def apply_threshold(
    y: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, Any]:
    accepted = probability >= threshold
    count = int(np.sum(accepted))
    if count == 0:
        return {"accepted": 0, "coverage": 0.0, "accuracy": None, "risk": None}
    accuracy = float(np.mean(y[accepted]))
    return {
        "accepted": count,
        "coverage": count / len(y),
        "accuracy": accuracy,
        "risk": 1.0 - accuracy,
    }


def percentile_interval(values: list[float]) -> list[float] | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(finite) == 0:
        return None
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def bootstrap_intervals(
    y: np.ndarray,
    probability: np.ndarray,
    baseline_probability: float,
    repetitions: int,
    seed: int,
) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {
        "accuracy": [],
        "mean_confidence": [],
        "calibration_gap": [],
        "brier_score": [],
        "brier_skill_score": [],
        "ece_equal_mass_5": [],
        "auroc": [],
    }
    for _ in range(repetitions):
        indices = rng.integers(0, len(y), size=len(y))
        metrics = metric_bundle(y[indices], probability[indices], baseline_probability)
        for name in collected:
            value = metrics[name]
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                collected[name].append(float(value))
    return {name: percentile_interval(values) for name, values in collected.items()}


def load_items(
    prompt_path: Path,
    confidence_path: Path,
    expected_condition: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts = index_unique(read_jsonl(prompt_path), f"{expected_condition} prompts")
    confidence_rows = index_unique(
        read_jsonl(confidence_path), f"{expected_condition} confidence results"
    )
    if set(prompts) != set(confidence_rows):
        missing = sorted(set(prompts) - set(confidence_rows))
        extra = sorted(set(confidence_rows) - set(prompts))
        raise ValueError(
            f"Confidence rows do not match prompts for {expected_condition}; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    items = []
    parse_errors: list[dict[str, Any]] = []
    for sample_id, prompt in prompts.items():
        if prompt.get("source_condition") != expected_condition:
            raise ValueError(f"Condition mismatch for {sample_id}")
        result = confidence_rows[sample_id]
        try:
            confidence = parse_confidence(str(result.get("response", result.get("prediction", ""))))
        except ValueError as exc:
            parse_errors.append(
                {
                    "id": sample_id,
                    "error": str(exc),
                    "finish_reason": result.get("finish_reason"),
                    "generated_token_count": result.get("generated_token_count"),
                    "reasoning_token_count": result.get("reasoning_token_count"),
                }
            )
            continue
        items.append(
            {
                "id": sample_id,
                "condition": expected_condition,
                "split": prompt["split"],
                "correct": bool(prompt["semantic_correct"]),
                "confidence": confidence,
                "source_prediction": prompt["source_prediction"],
                "response": result.get("response", ""),
            }
        )
    return items, parse_errors


def arrays(items: list[dict[str, Any]], split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    selected = sorted(
        (item for item in items if item["split"] == split), key=lambda item: item["id"]
    )
    return (
        [item["id"] for item in selected],
        np.asarray([item["correct"] for item in selected], dtype=int),
        np.asarray([item["confidence"] for item in selected], dtype=float),
    )


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "CRUXEval reflective-confidence calibration pilot",
        "=" * 49,
        "",
        "Correctness target: semantic Python value equivalence.",
        "Calibration and test IDs are paired across reasoning conditions.",
        (
            "Confidence extraction: "
            + ", ".join(
                f"{condition}={summary['confidence_extraction'][condition]['parsed']}/"
                f"{summary['confidence_extraction'][condition]['total']}"
                for condition in CONDITIONS
            )
            + f"; paired complete cases={summary['paired_complete_cases']}"
        ),
        "",
        "Raw held-out confidence",
        "-----------------------",
    ]
    for condition in CONDITIONS:
        metrics = summary["raw_test"][condition]["metrics"]
        lines.append(
            f"{condition:>4}: n={metrics['n']}, accuracy={fmt(metrics['accuracy'])}, "
            f"mean_confidence={fmt(metrics['mean_confidence'])}, "
            f"gap={fmt(metrics['calibration_gap'])}, AUROC={fmt(metrics['auroc'])}, "
            f"Brier={fmt(metrics['brier_score'])}, "
            f"BrierSkill={fmt(metrics['brier_skill_score'])}, "
            f"ECE5={fmt(metrics['ece_equal_mass_5'])}"
        )

    lines.extend(["", "Platt-scaled held-out and transfer evaluations", "----------------------------------------------"])
    for key, evaluation in summary["platt_evaluations"].items():
        metrics = evaluation["metrics"]
        lines.append(
            f"{key:>10}: accuracy={fmt(metrics['accuracy'])}, "
            f"mean_confidence={fmt(metrics['mean_confidence'])}, "
            f"gap={fmt(metrics['calibration_gap'])}, AUROC={fmt(metrics['auroc'])}, "
            f"Brier={fmt(metrics['brier_score'])}, "
            f"BrierSkill={fmt(metrics['brier_skill_score'])}, "
            f"ECE5={fmt(metrics['ece_equal_mass_5'])}"
        )

    lines.extend(["", "Raw-confidence risk/coverage on the held-out set", "-----------------------------------------------"])
    for condition in CONDITIONS:
        lines.append(f"{condition}:")
        for row in summary["raw_test"][condition]["risk_coverage"]:
            lines.append(
                f"  coverage={row['actual_coverage']:.2f}: "
                f"accuracy={row['accuracy']:.3f}, risk={row['risk']:.3f}"
            )

    lines.extend(["", "Threshold transfer", "------------------"])
    for source, threshold_result in summary["threshold_transfer"].items():
        selected = threshold_result["selected_on_calibration"]
        if selected is None:
            lines.append(f"{source}: no threshold met the calibration criterion")
            continue
        lines.append(
            f"{source}: selected threshold={selected['threshold']:.2f}, "
            f"calibration accuracy={selected['accuracy']:.3f}, "
            f"coverage={selected['coverage']:.3f}"
        )
        for target, result in threshold_result["applied_to_test"].items():
            lines.append(
                f"  -> {target}: accepted={result['accepted']}, "
                f"coverage={result['coverage']:.3f}, accuracy={fmt(result['accuracy'])}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstraps <= 0:
        raise ValueError("--bootstraps must be positive")
    if args.minimum_threshold_items <= 0:
        raise ValueError("--minimum-threshold-items must be positive")
    if not 0 < args.target_accuracy <= 1:
        raise ValueError("--target-accuracy must be in (0, 1]")

    prompt_paths = {
        "none": resolve_path(args.none_prompts),
        "low": resolve_path(args.low_prompts),
    }
    confidence_paths = {
        "none": resolve_path(args.none_confidence),
        "low": resolve_path(args.low_confidence),
    }
    loaded = {
        condition: load_items(prompt_paths[condition], confidence_paths[condition], condition)
        for condition in CONDITIONS
    }
    items = {condition: loaded[condition][0] for condition in CONDITIONS}
    parse_errors = {condition: loaded[condition][1] for condition in CONDITIONS}
    parsed_ids = {
        condition: {item["id"] for item in items[condition]} for condition in CONDITIONS
    }
    common_ids = set.intersection(*(parsed_ids[condition] for condition in CONDITIONS))
    items = {
        condition: [item for item in items[condition] if item["id"] in common_ids]
        for condition in CONDITIONS
    }
    data: dict[str, dict[str, tuple[list[str], np.ndarray, np.ndarray]]] = {
        condition: {
            split: arrays(items[condition], split) for split in ("calibration", "test")
        }
        for condition in CONDITIONS
    }
    for split in ("calibration", "test"):
        if data["none"][split][0] != data["low"][split][0]:
            raise ValueError(f"The {split} IDs are not paired across conditions")

    calibration_prevalence = {
        condition: float(np.mean(data[condition]["calibration"][1]))
        for condition in CONDITIONS
    }
    platt_models = {
        condition: fit_platt(
            data[condition]["calibration"][2], data[condition]["calibration"][1]
        )
        for condition in CONDITIONS
    }

    raw_test: dict[str, Any] = {}
    for index, condition in enumerate(CONDITIONS):
        _, y_test, probability_test = data[condition]["test"]
        metrics = metric_bundle(
            y_test, probability_test, calibration_prevalence[condition]
        )
        raw_test[condition] = {
            "metrics": metrics,
            "bootstrap_95_ci": bootstrap_intervals(
                y_test,
                probability_test,
                calibration_prevalence[condition],
                args.bootstraps,
                args.seed + index,
            ),
            "risk_coverage": risk_coverage(y_test, probability_test),
        }

    platt_evaluations: dict[str, Any] = {}
    evaluation_index = 10
    for source in CONDITIONS:
        for target in CONDITIONS:
            _, y_test, target_probability = data[target]["test"]
            calibrated = apply_platt(target_probability, platt_models[source])
            key = f"{source}->{target}"
            platt_evaluations[key] = {
                "source_calibrator": source,
                "target_condition": target,
                "metrics": metric_bundle(
                    y_test, calibrated, calibration_prevalence[source]
                ),
                "bootstrap_95_ci": bootstrap_intervals(
                    y_test,
                    calibrated,
                    calibration_prevalence[source],
                    args.bootstraps,
                    args.seed + evaluation_index,
                ),
            }
            evaluation_index += 1

    threshold_transfer: dict[str, Any] = {}
    for source in CONDITIONS:
        _, source_calibration_y, source_calibration_probability = data[source]["calibration"]
        selected = select_accuracy_threshold(
            source_calibration_y,
            source_calibration_probability,
            args.target_accuracy,
            args.minimum_threshold_items,
        )
        applied: dict[str, Any] = {}
        if selected is not None:
            for target in CONDITIONS:
                _, target_y, target_probability = data[target]["test"]
                applied[target] = apply_threshold(
                    target_y, target_probability, float(selected["threshold"])
                )
        threshold_transfer[source] = {
            "selected_on_calibration": selected,
            "applied_to_test": applied,
        }

    manifest_path = resolve_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {
        "pilot": "cruxeval_reflective_confidence_v1",
        "correctness_target": manifest["correctness_target"],
        "split_policy": manifest["split_policy"],
        "bootstraps": args.bootstraps,
        "bootstrap_seed": args.seed,
        "confidence_prompt_paths": {key: str(value) for key, value in prompt_paths.items()},
        "confidence_result_paths": {key: str(value) for key, value in confidence_paths.items()},
        "confidence_extraction": {
            condition: {
                "total": len(loaded[condition][0]) + len(parse_errors[condition]),
                "parsed": len(loaded[condition][0]),
                "failed": len(parse_errors[condition]),
                "failures": parse_errors[condition],
                "excluded_for_pairing": sorted(parsed_ids[condition] - common_ids),
            }
            for condition in CONDITIONS
        },
        "paired_complete_cases": len(common_ids),
        "split_sizes": {
            condition: {
                split: len(data[condition][split][0])
                for split in ("calibration", "test")
            }
            for condition in CONDITIONS
        },
        "calibration_prevalence": calibration_prevalence,
        "confidence_unique_values": {
            condition: int(len(np.unique(np.asarray([item["confidence"] for item in items[condition]]))))
            for condition in CONDITIONS
        },
        "platt_models": platt_models,
        "raw_test": raw_test,
        "platt_evaluations": platt_evaluations,
        "threshold_transfer": threshold_transfer,
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
