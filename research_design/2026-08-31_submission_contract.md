# FSE 2027 submission contract

Date: 2026-08-31
Target deadline: 2026-10-02

## Working thesis

Program-behavior accuracy does not determine whether individual predictions can
be trusted. Reliability depends on correctness discrimination, probability
calibration, selective risk, and transport across evaluation environments.

The study evaluates configured generator, verifier, benchmark, prompt, scoring,
and calibration systems. It does not claim that confidence is an intrinsic
property of a model or that correct outputs establish faithful internal
execution.

## Research questions

### RQ1: Reliability estimation

How well can post-hoc estimators distinguish correct from incorrect
program-behavior predictions, and what information produces the signal?

Primary comparisons:

- prevalence-only reference;
- response-only confidence;
- program-feature model;
- independent re-solve agreement;
- code-aware confidence.

### RQ2: Selective prediction

Under what generator, verifier, and inference conditions can reliability
estimates support useful abstention?

Primary outcomes:

- correctness and completion;
- AUROC and Brier skill;
- calibration intercept and slope;
- risk-coverage and selective accuracy;
- paired information-condition effects.

### RQ3: Reliability transport

How does a reliability mapping change across benchmarks and inference
conditions, and how much target information is required to restore calibration?

Primary comparisons:

- CRUXEval to RE2-Bench and the reverse direction;
- reasoning-condition transport;
- unchanged source calibrator;
- zero-label prior correction;
- target-intercept updates with increasing labeled sample sizes;
- full target refit.

## Completed empirical core

1. CRUXEval: all 800 output-prediction items, GPT-5.4 Nano source reasoning
   `none` and `low`, code-aware and response-only confidence.
2. RE2-Bench Lite: all 500 items, GPT-5.4 Nano source reasoning `none` and
   `low`, paired response-only and code-aware verification.
3. CRUXEval to RE2 calibration transport under the comparable low-reasoning
   source and verifier configuration.
4. Deterministic execution-reference scoring and scoring-policy sensitivity.

## Required remaining empirical work

1. Agreement baseline from existing independent predictions.
2. Out-of-sample program-feature baseline.
3. Reverse transport, zero-label prior correction, and label-efficiency curves.
4. One open-weight generator and verifier replication.
5. Cross-family generator-by-verifier cells on RE2-Bench.
6. Test-oracle decision analysis using the existing risk-coverage results.

## Model scope

Mandatory:

- GPT-5.4 Nano, already complete;
- one qualified open-weight model, currently Qwen3.6 27B.

Supporting only:

- DeepSeek V4 Flash reasoning-setting experiment.

No third full model family is added before the analysis freeze.

## Paper exclusions

The following do not enter the main argument:

- synthetic V1, V2, or V3;
- red-black-tree experiments;
- R vignette experiments;
- broad Qwen parameter scaling;
- the full benchmark-claim audit framework;
- change-impact and execution-oracle pilots.

## Statistical rules

- Use deterministic execution-derived correctness labels.
- Use paired item bootstrap intervals for paired interventions.
- Report operational handling of parse failures and truncations.
- Treat ECE as descriptive, not as the sole calibration measure.
- Separate discrimination from calibration.
- Do not compare vendor reasoning-effort labels as equivalent computation.
- Do not interpret observational benchmark strata causally.

## Dated gates

- 2026-09-02: freeze RQs, metrics, baselines, artifact boundaries, and manuscript skeleton.
- 2026-09-05: open-weight pilot must complete end to end with acceptable route and parse checks.
- 2026-09-10: if the full open-weight experiment is not scored, demote it to a pilot.
- 2026-09-13: no new experiment families.
- 2026-09-15: analysis freeze.
- 2026-09-20: complete manuscript draft freeze.
- 2026-09-21 through 2026-09-24: internal review.
- 2026-09-25 through 2026-09-28: revision and artifact validation.
- 2026-09-29 through 2026-10-01: final submission checks.

## Repository policy

Licensing-sensitive benchmark content and raw inference records remain under
`_local/` until redistribution rights are resolved. The public artifact may
include analysis code, configurations, hashes, aggregate results, and only
those derived records whose redistribution is confirmed.

