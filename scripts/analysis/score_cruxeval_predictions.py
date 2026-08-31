"""Create stable item-level scores from CRUXEval model predictions."""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from cruxeval_taxonomy import (
    EVAL_FAIL,
    categorize_error,
    literal_eval_safe,
    literal_eval_type,
    string_residual_analysis,
    target_type_name,
    type_error_category,
)
from semantic_scoring import score_python_literal
from utils import clean_prediction, read_jsonl_with_errors, write_jsonl


DEFAULT_DATASET_PATH = Path("data/raw/cruxeval/cruxeval.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CRUXEval prediction JSONL.")
    parser.add_argument("predictions", type=Path, help="Prediction JSONL to score.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, required=True, help="Item-level score JSONL.")
    return parser.parse_args()


def read_dataset(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return {row["id"]: row for row in csv.DictReader(csv_file)}


def strict_match(prediction: str, target: str) -> bool:
    return clean_prediction(prediction) == str(target).strip()


def relaxed_match(prediction: str, target: str) -> bool:
    prediction_clean = clean_prediction(prediction)
    target_clean = str(target).strip()
    if prediction_clean == target_clean:
        return True

    target_value = literal_eval_safe(target_clean)
    prediction_value = literal_eval_safe(prediction_clean)
    if target_value is not EVAL_FAIL and prediction_value is not EVAL_FAIL:
        return type(prediction_value) is type(target_value) and prediction_value == target_value
    if isinstance(target_value, str):
        return prediction_clean == target_value
    return False


def bucket(value: int, bounds: list[int]) -> str:
    previous = 0
    for bound in bounds:
        if value <= bound:
            return f"<= {bound}" if previous == 0 else f"{previous + 1}-{bound}"
        previous = bound
    return f"> {bounds[-1]}"


def code_features(code: str) -> list[str]:
    checks = {
        "if": "if " in code,
        "for": "for " in code,
        "while": "while " in code,
        "append": ".append" in code,
        "sort": ".sort" in code or "sorted(" in code,
        "dict": "dict" in code or "{}" in code,
        "set": "set(" in code or ".add" in code,
        "lambda": "lambda" in code,
    }
    features = [name for name, present in checks.items() if present]
    return features or ["none_of_tracked"]


def residual_fields(prefix: str, residual: dict[str, Any] | None) -> dict[str, Any]:
    names = (
        "input_relation",
        "edit_shape",
        "pred_target_distance",
        "pred_input_distance",
        "input_target_distance",
        "nearest_input",
        "target_distance_band",
    )
    return {f"{prefix}_string_{name}": residual.get(name) if residual else None for name in names}


def score_record(record: dict[str, Any], dataset_row: dict[str, str]) -> dict[str, Any]:
    prediction = record.get("prediction", "")
    prediction_clean = clean_prediction(prediction)
    target = record.get("target_output", dataset_row["output"])
    strict_correct = strict_match(prediction, target)
    relaxed_correct = relaxed_match(prediction, target)
    semantic_score = score_python_literal(
        prediction,
        target,
        finish_reason=record.get("finish_reason", ""),
        native_finish_reason=record.get("native_finish_reason", ""),
    )

    parsed_target, target_error = literal_eval_type(target)
    parsed_prediction, prediction_error = literal_eval_type(prediction_clean)
    target_type = target_type_name(parsed_target, target_error)
    broad_error = categorize_error(parsed_prediction, prediction_error, parsed_target, target_error)

    strict_type_error = type_error_category(
        parsed_target,
        target_error,
        parsed_prediction,
        prediction_error,
        prediction_clean,
        strict_correct,
    )
    relaxed_type_error = type_error_category(
        parsed_target,
        target_error,
        parsed_prediction,
        prediction_error,
        prediction_clean,
        relaxed_correct,
    )
    strict_residual = string_residual_analysis(
        parsed_target, parsed_prediction, prediction_clean, dataset_row["input"], strict_type_error
    )
    relaxed_residual = string_residual_analysis(
        parsed_target, parsed_prediction, prediction_clean, dataset_row["input"], relaxed_type_error
    )

    code = dataset_row["code"]
    input_text = dataset_row["input"]
    dataset_target = dataset_row["output"]
    response = str(record.get("response", ""))
    features = code_features(code)
    scored = {
        "id": record.get("id", ""),
        "unknown_id": False,
        "model": record.get("model", ""),
        "prompt_version": record.get("prompt_version", ""),
        "strict_correct": strict_correct,
        "relaxed_correct": relaxed_correct,
        "typed_semantic_correct": semantic_score.typed_semantic_correct,
        "bare_string_content_correct": semantic_score.bare_string_content_correct,
        "semantic_status": semantic_score.semantic_status,
        "prediction_parseable": semantic_score.prediction_parseable,
        "target_parseable": semantic_score.target_parseable,
        "is_truncated": semantic_score.is_truncated,
        "has_extracted_answer": semantic_score.has_extracted_answer,
        "relaxed_only_correct": relaxed_correct and not strict_correct,
        "stored_correct": record.get("correct") if "correct" in record else None,
        "target_matches_dataset_strict": strict_match(target, dataset_target),
        "target_matches_dataset_relaxed": relaxed_match(target, dataset_target),
        "finish_reason": record.get("finish_reason", ""),
        "generated_token_count": record.get("generated_token_count"),
        "empty_prediction": not str(prediction).strip(),
        "response_starts_final_output": response.strip().lower().startswith("final_output:"),
        "response_is_multiline": "\n" in response.strip(),
        "target_type": target_type,
        "dataset_family": dataset_row.get("family"),
        "dataset_subfamily": dataset_row.get("subfamily"),
        "dataset_difficulty": dataset_row.get("difficulty"),
        "dataset_trace_events": dataset_row.get("trace_events"),
        "dataset_params": dataset_row.get("params"),
        "strict_top_error_category": "correct" if strict_correct else broad_error,
        "relaxed_top_error_category": "correct" if relaxed_correct else broad_error,
        "strict_type_error_category": strict_type_error,
        "relaxed_type_error_category": relaxed_type_error,
        "features": features,
        "code_first_line": code.splitlines()[0] if code.splitlines() else "",
        "code_chars": len(code),
        "code_lines": len(code.splitlines()),
        "input_chars": len(input_text),
        "target_chars": len(dataset_target),
        "prediction_chars": len(prediction_clean),
        "code_chars_bucket": bucket(len(code), [80, 160, 240, 400]),
        "code_lines_bucket": bucket(len(code.splitlines()), [3, 5, 8, 12]),
        "input_chars_bucket": bucket(len(input_text), [10, 30, 80, 160]),
        "target_chars_bucket": bucket(len(dataset_target), [5, 15, 40, 100]),
        "prediction": prediction,
        "prediction_clean": prediction_clean,
        "target_output": target,
    }
    scored.update(residual_fields("strict", strict_residual))
    scored.update(residual_fields("relaxed", relaxed_residual))
    return scored


def main() -> None:
    args = parse_args()
    dataset = read_dataset(args.dataset)
    predictions, json_errors = read_jsonl_with_errors(args.predictions)
    if not predictions:
        raise RuntimeError(f"No prediction records found in {args.predictions}")

    prediction_ids = [str(record.get("id", "")) for record in predictions]
    duplicate_ids = sorted(sample_id for sample_id, count in Counter(prediction_ids).items() if count > 1)
    missing_ids = sorted(set(dataset) - set(prediction_ids))
    unknown_ids = sorted(set(prediction_ids) - set(dataset))

    scored_rows = []
    for record in predictions:
        sample_id = str(record.get("id", ""))
        dataset_row = dataset.get(sample_id)
        if dataset_row is None:
            scored_rows.append(
                {
                    "id": sample_id,
                    "unknown_id": True,
                    "model": record.get("model", ""),
                    "prompt_version": record.get("prompt_version", ""),
                    "finish_reason": record.get("finish_reason", ""),
                    "generated_token_count": record.get("generated_token_count"),
                    "prediction": record.get("prediction", ""),
                    "target_output": record.get("target_output", ""),
                }
            )
            continue
        scored_rows.append(score_record(record, dataset_row))

    write_jsonl(args.output, scored_rows)
    known_rows = [row for row in scored_rows if not row["unknown_id"]]
    print(f"Predictions:    {args.predictions}")
    print(f"Dataset:        {args.dataset}")
    print(f"Scored rows:    {len(known_rows)}")
    print(f"JSON errors:    {len(json_errors)}")
    print(f"Duplicate IDs:  {len(duplicate_ids)}")
    print(f"Missing IDs:    {len(missing_ids)}")
    print(f"Unknown IDs:    {len(unknown_ids)}")
    print(f"Strict correct: {sum(row['strict_correct'] for row in known_rows)}")
    print(f"Relaxed correct:{sum(row['relaxed_correct'] for row in known_rows):>4}")
    print(f"Wrote item-level scores to {args.output.resolve()}")

    if json_errors:
        print(f"Malformed JSON lines: {json_errors[:3]}")
    if duplicate_ids:
        print(f"Duplicate ID examples: {duplicate_ids[:3]}")
    if missing_ids:
        print(f"Missing ID examples: {missing_ids[:3]}")
    if unknown_ids:
        print(f"Unknown ID examples: {unknown_ids[:3]}")


if __name__ == "__main__":
    main()
