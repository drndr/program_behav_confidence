"""Decompose calibration transport from CRUXEval to full RE2-Bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_cruxeval_calibration_deconfounders import (  # noqa: E402
    apply_calibration,
    bootstrap_transfer_pipeline,
    evaluated_metrics,
    fit_calibration_model,
)
from analyze_cruxeval_calibration_pilot import parse_confidence  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFORMATION_CONDITIONS = ("response_only", "code_aware")
DEFAULT_BOOTSTRAPS = 5000
DEFAULT_SEED = 20260819


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-freeze",
        type=Path,
        default=Path("research_design/2026-08-19_re2_calibration_transfer_split.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "research_design/2026-08-19_cruxeval_re2_calibration_transfer_analysis.json"
        ),
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=Path(
            "research_design/2026-08-19_cruxeval_re2_calibration_transfer_results.txt"
        ),
    )
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        rows = [json.loads(line) for line in input_file if line.strip()]
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in indexed:
            raise ValueError(f"Missing or duplicate ID in {label}: {sample_id!r}")
        indexed[sample_id] = row
    return indexed


def crux_paths(information_condition: str) -> tuple[Path, Path]:
    if information_condition == "code_aware":
        return (
            PROJECT_ROOT / "data/prompts/cruxeval/cruxeval_reflective_confidence_low_v1.jsonl",
            PROJECT_ROOT
            / "results/cruxeval_reflective_confidence_gpt_5_4_nano_low_predictions.jsonl",
        )
    return (
        PROJECT_ROOT
        / "data/prompts/cruxeval/cruxeval_prediction_only_confidence_low_v1.jsonl",
        PROJECT_ROOT
        / "results/cruxeval_prediction_only_confidence_gpt_5_4_nano_low_predictions.jsonl",
    )


def re2_paths(information_condition: str) -> tuple[Path, Path]:
    return (
        PROJECT_ROOT
        / f"tmp/re2_bench_full_confidence_low_{information_condition}_v1.jsonl",
        PROJECT_ROOT
        / f"tmp/re2_bench_full_confidence_low_{information_condition}_predictions.jsonl",
    )


def load_crux(information_condition: str) -> list[dict[str, Any]]:
    prompt_path, result_path = crux_paths(information_condition)
    prompts = index_unique(read_jsonl(prompt_path), str(prompt_path))
    results = index_unique(read_jsonl(result_path), str(result_path))
    if set(prompts) != set(results) or len(prompts) != 800:
        raise ValueError("CRUXEval confidence prompts/results must contain the same 800 IDs")
    rows = []
    for sample_id in sorted(prompts):
        prompt = prompts[sample_id]
        result = results[sample_id]
        try:
            confidence = parse_confidence(str(result.get("response", "")))
            parse_failure = False
        except ValueError:
            confidence = 0.0
            parse_failure = True
        rows.append(
            {
                "id": sample_id,
                "split": str(prompt["split"]),
                "correct": bool(prompt["semantic_correct"]),
                "confidence": confidence,
                "parse_failure": parse_failure,
            }
        )
    return rows


def load_re2(
    information_condition: str, assignments: dict[str, str]
) -> list[dict[str, Any]]:
    prompt_path, result_path = re2_paths(information_condition)
    prompts = index_unique(read_jsonl(prompt_path), str(prompt_path))
    results = index_unique(read_jsonl(result_path), str(result_path))
    if set(prompts) != set(results) or set(prompts) != set(assignments):
        raise ValueError("RE2 prompts, results, and frozen assignments must match")
    rows = []
    for sample_id in sorted(prompts):
        prompt = prompts[sample_id]
        result = results[sample_id]
        try:
            confidence = parse_confidence(str(result.get("response", "")))
            parse_failure = False
        except ValueError:
            confidence = 0.0
            parse_failure = True
        rows.append(
            {
                "id": sample_id,
                "split": assignments[sample_id],
                "correct": bool(prompt["safe_compatibility_correct"]),
                "confidence": confidence,
                "parse_failure": parse_failure,
            }
        )
    return rows


def arrays(rows: list[dict[str, Any]], split: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["split"] == split]
    return (
        np.asarray([row["correct"] for row in selected], dtype=int),
        np.asarray([row["confidence"] for row in selected], dtype=float),
    )


def decompose(
    crux_rows: list[dict[str, Any]],
    re2_rows: list[dict[str, Any]],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    source_calibration_y, source_calibration_probability = arrays(
        crux_rows, "calibration"
    )
    source_test_y, source_test_probability = arrays(crux_rows, "test")
    target_calibration_y, target_calibration_probability = arrays(
        re2_rows, "calibration"
    )
    target_test_y, target_test_probability = arrays(re2_rows, "test")
    source_prevalence = float(np.mean(source_calibration_y))
    target_prevalence = float(np.mean(target_calibration_y))

    source_model = fit_calibration_model(
        source_calibration_probability, source_calibration_y
    )
    source_slope_target_intercept = fit_calibration_model(
        target_calibration_probability,
        target_calibration_y,
        fixed_slope=float(source_model["slope"]),
    )
    target_model = fit_calibration_model(
        target_calibration_probability, target_calibration_y
    )

    source_test_calibrated = apply_calibration(source_test_probability, source_model)
    target_uncorrected = apply_calibration(target_test_probability, source_model)
    target_intercept_updated = apply_calibration(
        target_test_probability, source_slope_target_intercept
    )
    target_refit = apply_calibration(target_test_probability, target_model)
    likelihood_ratio = max(
        0.0,
        2
        * (
            float(source_slope_target_intercept["negative_log_likelihood"])
            - float(target_model["negative_log_likelihood"])
        ),
    )
    bootstrap = bootstrap_transfer_pipeline(
        source_calibration_y,
        source_calibration_probability,
        target_calibration_y,
        target_calibration_probability,
        target_test_y,
        target_test_probability,
        repetitions,
        seed,
    )
    intercept_metrics = evaluated_metrics(
        target_test_y, target_intercept_updated, target_prevalence
    )
    target_metrics = evaluated_metrics(target_test_y, target_refit, target_prevalence)
    bootstrap.update(
        {
            "brier_difference_point": (
                intercept_metrics["brier_score"] - target_metrics["brier_score"]
            ),
            "log_loss_difference_point": (
                intercept_metrics["log_loss"] - target_metrics["log_loss"]
            ),
        }
    )
    return {
        "source_calibration_n": len(source_calibration_y),
        "source_test_n": len(source_test_y),
        "target_calibration_n": len(target_calibration_y),
        "target_test_n": len(target_test_y),
        "source_calibration_prevalence": source_prevalence,
        "target_calibration_prevalence": target_prevalence,
        "target_test_prevalence": float(np.mean(target_test_y)),
        "source_full_model": source_model,
        "source_slope_target_intercept_model": source_slope_target_intercept,
        "target_full_model": target_model,
        "slope_likelihood_ratio": {
            "statistic": likelihood_ratio,
            "p_value": float(chi2.sf(likelihood_ratio, 1)),
            "null": "RE2 calibration slope equals the CRUXEval calibration slope",
        },
        "source_cruxeval_test": {
            "raw": evaluated_metrics(
                source_test_y, source_test_probability, source_prevalence
            ),
            "source_calibrated": evaluated_metrics(
                source_test_y, source_test_calibrated, source_prevalence
            ),
        },
        "held_out_re2_test": {
            "raw": evaluated_metrics(
                target_test_y, target_test_probability, target_prevalence
            ),
            "cruxeval_full_uncorrected": evaluated_metrics(
                target_test_y, target_uncorrected, target_prevalence
            ),
            "cruxeval_slope_re2_intercept": intercept_metrics,
            "re2_full_refit": target_metrics,
        },
        "residual_slope_penalty": bootstrap,
        "re2_operational_confidence_failures": sum(
            bool(row.get("parse_failure")) for row in re2_rows
        ),
        "cruxeval_operational_confidence_failures": sum(
            bool(row.get("parse_failure")) for row in crux_rows
        ),
    }


def fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "CRUXEval to RE2 calibration transport",
        "=" * 38,
        "",
        "Primary comparable setting: GPT-5.4 Nano low-reasoning source predictions and a low-reasoning verifier.",
        "RE2 confidence failures receive operational confidence zero.",
        "",
    ]
    for information_condition in INFORMATION_CONDITIONS:
        result = summary["information_conditions"][information_condition]
        held_out = result["held_out_re2_test"]
        penalty = result["residual_slope_penalty"]
        interval = penalty["brier_difference_95_ci"]
        lines.extend(
            [
                information_condition,
                "-" * len(information_condition),
                (
                    f"Prevalence: CRUX calibration={result['source_calibration_prevalence']:.3f}; "
                    f"RE2 calibration={result['target_calibration_prevalence']:.3f}; "
                    f"RE2 test={result['target_test_prevalence']:.3f}."
                ),
                (
                    f"Slopes: CRUX={result['source_full_model']['slope']:.3f}; "
                    f"RE2={result['target_full_model']['slope']:.3f}; "
                    f"LR p={result['slope_likelihood_ratio']['p_value']:.3g}."
                ),
                (
                    f"RE2 raw: gap={held_out['raw']['calibration_gap']:+.3f}, "
                    f"Brier={held_out['raw']['brier_score']:.3f}."
                ),
                (
                    f"CRUX calibrator unchanged: gap={held_out['cruxeval_full_uncorrected']['calibration_gap']:+.3f}, "
                    f"Brier={held_out['cruxeval_full_uncorrected']['brier_score']:.3f}."
                ),
                (
                    f"After RE2 intercept update: gap={held_out['cruxeval_slope_re2_intercept']['calibration_gap']:+.3f}, "
                    f"Brier={held_out['cruxeval_slope_re2_intercept']['brier_score']:.3f}."
                ),
                (
                    f"RE2 full refit: gap={held_out['re2_full_refit']['calibration_gap']:+.3f}, "
                    f"Brier={held_out['re2_full_refit']['brier_score']:.3f}."
                ),
                (
                    f"Residual Brier penalty from retaining the CRUX slope: "
                    f"{penalty['brier_difference_point']:+.3f} "
                    f"(95% CI {interval[0]:+.3f} to {interval[1]:+.3f})."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstraps <= 0:
        raise ValueError("--bootstraps must be positive")
    split_path = resolve_path(args.split_freeze)
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = {
        str(sample_id): str(split)
        for sample_id, split in split_payload["assignments"].items()
    }
    results = {}
    for index, information_condition in enumerate(INFORMATION_CONDITIONS):
        crux_rows = load_crux(information_condition)
        re2_rows = load_re2(information_condition, assignments)
        results[information_condition] = decompose(
            crux_rows,
            re2_rows,
            args.bootstraps,
            args.seed + index,
        )
    summary = {
        "analysis": "cruxeval_to_re2_calibration_transport_v1",
        "status": "prespecified_transport_family_with_split_frozen_before_computation",
        "primary_setting": (
            "GPT-5.4 Nano low source reasoning and low verifier reasoning in both benchmarks"
        ),
        "correctness_contract_note": (
            "CRUXEval uses relaxed Python value equivalence; RE2 uses safe compatibility "
            "system success, so the transport target intentionally includes benchmark ecology."
        ),
        "split_freeze_path": str(split_path),
        "bootstraps": args.bootstraps,
        "bootstrap_seed": args.seed,
        "information_conditions": results,
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
