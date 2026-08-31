# Full RE2-Bench calibration protocol

Status: frozen before new full-corpus inference.

## Objective

Test whether a model can discriminate reliable from unreliable program-output
predictions on the complete released 500-item RE2-Bench Lite manifest, and
whether access to the program and input adds information beyond the proposed
response alone.

## Corpus

- Pinned RE2-Bench commit: `57b971f18567c879a7180bb6bd5b8e544dd9a47e`
- Prompt artifact: `tmp/re2_bench_lite_output_prediction_prompts.jsonl`
- Population: all 500 released items, comprising 250 HC and 250 LC items
- The previously selected 132-item Real/Swebench holdout remains a prespecified
  matched-source sensitivity subset. It is not the primary population.

The complete corpus is used for the benchmark-level estimate. Complexity
comparisons on the full corpus are descriptive because complexity and source
dataset are associated. The matched 132-item subset is used to check whether
the main conclusions survive source matching.

## Source predictions

Model and route are fixed to `openai/gpt-5.4-nano-20260317` through the OpenAI
provider on OpenRouter. The two source conditions differ only in provider-native
reasoning effort:

- `none`, maximum 4,096 generated tokens, seed 0
- `low`, maximum 4,096 generated tokens, seed 0

The upstream RE2 prompt is retained exactly. It requests a visible reasoning
trace, so `none` denotes no provider-native hidden reasoning effort, not the
absence of a prompted rationale.

Previously collected low-reasoning responses are not imported into the new
full-corpus checkpoint because the earlier factorial runner used a different
prompt-file hash and schedule. The full 500-item runs form clean, independently
auditable checkpoints.

## Confidence judgments

Confidence is elicited after source generation using the same fixed verifier
for every candidate: GPT-5.4 Nano with low provider-native reasoning effort,
seed 0, and structured integer confidence from 0 to 100.

Each source candidate is judged in two paired information conditions:

1. `response_only`: the candidate's final response is shown, but the program
   and input are withheld.
2. `code_aware`: the identical candidate response is shown together with the
   original RE2 program, input, output structure, and task instructions.

The target is system success under the existing `safe_compatibility_correct`
policy, including output extraction and parsing. The ground-truth output is
never included in an inference message.

Holding verifier effort fixed prevents source reasoning effort from being
confounded with verifier capability.

## Outcomes

Primary outcome:

- Paired difference in AUROC between `code_aware` and `response_only`, analyzed
  separately for `none` and `low` source predictions, with item bootstrap
  confidence intervals.

Primary correctness policy:

- `safe_compatibility_correct` from the existing safe RE2 scorer.

Secondary outcomes:

- Brier score, log loss, calibration slope/intercept, and ECE
- selective accuracy/risk as coverage decreases
- difference between source reasoning conditions under a fixed verifier
- calibration transport from CRUXEval to RE2, decomposing intercept/prior shift
  from slope/shape change
- results by source dataset, complexity group, and target type
- matched 132-item Real/Swebench sensitivity analysis
- strict and repaired typed-scoring sensitivity analyses

No HC-versus-LC contrast on the complete corpus will be described as causal.
Subgroup findings are exploratory unless separately powered and corrected.

## Missingness and failures

API errors and truncated generations count as system failures in the primary
analysis. Confidence parse failures are reported and are not silently dropped.
All exclusions, retries, route metadata, token usage, and reported costs are
retained.

## Redistribution

The pinned upstream repository had no detected license file when imported.
Prompts, predictions, and derived item-level artifacts remain under ignored
`tmp/` storage until redistribution rights are clarified. Code, aggregate
statistics, hashes, and protocols may be committed.
