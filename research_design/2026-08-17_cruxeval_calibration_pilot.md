# CRUXEval reflective-confidence calibration pilot

## Question

Can GPT-5.4 Nano identify which of its CRUXEval output predictions are correct, and does a
confidence mapping learned under one reasoning setting transfer to another?

## Fixed inputs

- Benchmark: all 800 CRUXEval output-prediction items already evaluated in this repository.
- Model and revision: `openai/gpt-5.4-nano`, revision
  `openai/gpt-5.4-nano-20260317`, routed only through OpenAI on OpenRouter.
- Source conditions: reasoning effort `none` and `low`.
- Correctness target: semantic Python value equivalence (`relaxed_correct` in the existing
  item-level scores). Canonical string equality is not the primary target because it treats
  equivalent Python literal formatting as incorrect.

## Confidence elicitation

Each existing prediction is shown to the same model together with the original program and
input. The ground-truth output and the original reasoning trace are not shown. The model returns
one integer confidence percentage under a strict JSON schema. Reflection uses the same reasoning
effort as the source prediction.

## Split

The 800 problem IDs are split once into 400 calibration and 400 test items. The assignment is
paired across reasoning conditions and stratified by the joint correctness pattern of the two
conditions. A seeded SHA-256 ordering makes the split deterministic.

## Analysis

For raw held-out confidence, report accuracy, mean confidence, calibration gap, five-bin
equal-mass ECE, Brier score, Brier skill against a calibration-prevalence predictor, AUROC, and
accuracy at 25%, 50%, 75%, and 100% coverage.

Risk-coverage points are threshold based. If the requested boundary falls inside a confidence tie,
all predictions with that confidence are accepted and the resulting actual coverage is reported.

Fit one Platt calibrator per reasoning condition on the calibration split. Evaluate each calibrator
on the held-out predictions from both reasoning conditions. This yields two within-condition and
two cross-condition evaluations.

On each calibration condition, select the raw-confidence threshold with greatest coverage among
thresholds attaining at least 90% empirical accuracy with at least 40 accepted items. Freeze the
threshold and apply it to both held-out reasoning conditions.

Use 2,000 bootstrap resamples of held-out problem IDs for 95% percentile intervals. ECE is
reported descriptively and is not treated as sufficient evidence of useful item-level confidence.

## Interpretation

- Positive Brier skill and AUROC above chance indicate useful item-level discrimination.
- Improved accuracy as coverage decreases indicates value for selective prediction.
- Good within-condition calibration but poor cross-condition calibration indicates calibration
  shift under reasoning effort.
- AUROC degradation across conditions indicates loss of discrimination, not merely a shifted
  probability scale.
- A threshold that misses its 90% target after transfer demonstrates failure of an operational
  reliability contract.
