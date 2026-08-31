# Manuscript outline

## 1. Introduction

- Program-behavior predictions as engineering oracles
- Why benchmark accuracy is not an individual reliability guarantee
- Distinction between discrimination, calibration, selective prediction, and transport
- Research questions and contributions

## 2. Background and related work

- Program-behavior prediction benchmarks
- Robustness, intermediate behavior, and reasoning analyses
- Confidence and calibration in code and language models
- Selective prediction and uncertainty under distribution shift
- Gap exposed by the concept matrix

## 3. Reliability framework

- Generator, candidate output, verifier, calibrator, deployment decision
- Correctness target and scoring policy
- Discrimination versus calibration
- Risk-coverage and abstention
- Calibration transport and prior shift

## 4. Study design

- CRUXEval and RE2-Bench populations
- Models, revisions, providers, prompts, and inference settings
- Information conditions and baselines
- Splits, estimands, uncertainty intervals, and missingness
- Reproducibility and licensing boundaries

## 5. RQ1: Reliability estimation

- Response-only versus code-aware confidence
- Program-feature baseline
- Independent re-solve agreement
- Generator-by-verifier comparison

## 6. RQ2: Selective prediction

- Accuracy and confidence by inference condition
- Calibration and Brier skill
- Risk-coverage curves
- Test-oracle decision analysis

## 7. RQ3: Reliability transport

- Cross-reasoning transport
- CRUXEval and RE2 transport in both directions
- Zero-label prior correction
- Few-label intercept update and full refit

## 8. Discussion

- What benchmark scores legitimately support
- When verification is re-solving rather than introspection
- Practical guidance for using predicted outputs
- Reporting recommendations

## 9. Threats to validity

- Benchmark and model scope
- Verifier and generator coupling
- Scoring-policy dependence
- Adaptive reasoning-token use
- Calibration-split reuse and exploratory analyses
- Artifact licensing

## 10. Conclusion

