# Literature forward search: output prediction × confidence × program properties

Date: 22 September 2026. Targeted forward-citation and keyword search; not a systematic
review. Absence from the results is not proof that no such work exists.

## Question

Is the intersection of (1) program output prediction, (2) confidence measurement and
calibration, and (3) program properties already occupied? And is reasoning effort
(trace length, reasoning tokens) as a confidence signal already studied for code
execution?

## Method

- **Forward citations (Semantic Scholar Graph API)** of 25 seed papers:
  - code-reasoning benchmarks: CRUXEval, REval, RE2-Bench, CodeMind, CodeSense,
    CodeCrash, ExeRScope, LiveCodeBench, CodeGlance, How Robustly Do LLMs Understand
    Execution Semantics, LM-CC, Demystifying Errors in LLM Reasoning Traces,
    What I cannot execute;
  - code confidence/calibration: Calibration and Correctness of LMs for Code (ICSE'25),
    Localized Calibrated Uncertainty in Code LMs, Multicalibration for LLM-based Code
    Generation, When to Answer and When to Defer (ICSE-NIER), Open the Oyster;
  - reasoning length: Trace Length is a Simple Uncertainty Signal, Revisiting the
    Test-Time Scaling of o1-like Models, Between Underthinking and Overthinking,
    Reasoning Models Better Express Their Confidence, An Empirical Study of Reasoning
    Steps in Thinking Code LLMs.
  - Result: 2,819 unique citing papers, 2,686 with abstracts.
- **Keyword search (Semantic Scholar bulk, boolean, 2024–2026)**: confidence × execution,
  effort × code, complexity × confidence, selective prediction × code, verifier ×
  output prediction. 368 papers not already in the citation pool; mostly noise
  ("output prediction" also matches power-grid forecasting).
- **Screening**: regular expressions over title and abstract for confidence,
  execution/output prediction, program properties and reasoning effort; ~110 flagged
  papers triaged by hand, ~20 abstracts read closely, full text read for the closest
  three (Playing Psychic, Thinking Hard Not Smart, What I cannot execute).
- **Coverage gaps**: OpenAlex has almost no reference data for arXiv preprints; the arXiv
  API was blocked from the search environment (HTTP 406); Semantic Scholar citation data
  lags for recent papers; work with unusual terminology or appearing only in proceedings
  may be missed. No Google Scholar.

## Findings

### 1. Output prediction × confidence × program properties: unoccupied

No paper found studies whether confidence tracks program properties in output
prediction. Papers citing Open the Oyster, Multicalibration or Gros & Devanbu do not
extend to execution × properties. Recent code-confidence work targets generated code:

- Code-MUE: uncertainty via execution-based semantic interaction graphs (arXiv 2607.12273)
- On the Robustness of LLMs' Internal Representation of Code Correctness (arXiv 2608.08266)
- Ensemble-Based Uncertainty Estimation for Code Correctness Estimation (arXiv 2603.27098)
- Task Abstention for LLMs in Code Generation (arXiv 2605.17029)
- Latent Programming Horizons in Coding Agents (arXiv 2607.05188)

Closest partial overlaps:

| Work | Output pred. | Confidence | Properties | Gap |
|---|---|---|---|---|
| Open the Oyster (2511.02197) | ✓ | ✓ | – | Compares task categories (output vs. state vs. path prediction), not program features |
| Armengol-Estapé et al., What I cannot execute (2503.05703) | ✓ | (✓) | (✓) | One figure: token NLL rises with "steps into the future"; one trained model; side analysis |
| Multicalibration for LLM-based Code Generation (2512.08810) | – | ✓ | ✓ | Code generation; complexity/length/language groups |
| CodeGlance, RE2-Bench, Spiess et al. 2026, LM-CC | ✓ | – | ✓ | No confidence measurement |

### 2. Reasoning effort as confidence: studied in general, and now in code execution

**Playing Psychic: Using Thought Trees to Predict Reasoning Models Accuracy on Coding
Tasks (arXiv 2604.16931, April 2026)** is the closest work to the effort angle.

- Tasks: CRUXEval output prediction, plus SAFIM (fill-in-the-middle) and CodeLingua
  (translation).
- Difficulty: CRUXEval functions composed as f₂(f₁(x)) at depths L1–L3 — a controllable
  program property.
- Finding: the structure of the reasoning trace predicts correctness; a random forest on
  thought-tree features reaches ~88–89% accuracy; length-only features underperform the
  full feature set; failure modes are tied to composition depth.
- Not covered: AUROC or calibration metrics; comparison with verbalized, verifier or
  token-probability confidence; naturally occurring real-world code; zero-shot signals
  (it trains a classifier). No citations yet (Semantic Scholar, September 2026).

**Thinking Hard, Not Smart: Reasoning Models Fail to Ration Test-Time Compute Across
Questions (arXiv 2608.07968, August 2026)** includes CRUXEval-O. Harder questions receive
more tokens, but reactively; the effect shrinks under tighter shared budgets. It does not
predict correctness from effort.

General (non-code) background:

- Devic et al., Trace Length is a Simple Uncertainty Signal in Reasoning Models
  (arXiv 2510.10409): trace length as zero-shot confidence, comparable to and
  complementary with verbalized confidence; emerges after reasoning post-training;
  survives adjustment for difficulty (GPT-4o-labelled DeepMath levels). Ten QA/multiple
  choice datasets, no code. Names difficulty measurement as a limitation.
- Zeng et al., Revisiting the Test-Time Scaling of o1-like Models (arXiv 2502.12215):
  correct solutions are often shorter than incorrect ones.
- Su et al., Between Underthinking and Overthinking (arXiv 2505.00127): models overthink
  easy and underthink hard problems.
- Yoon et al., Reasoning Models Better Express Their Confidence (arXiv 2505.14489,
  NeurIPS 2025).
- Lexical hints of accuracy in LLM reasoning chains (arXiv 2508.15842, Scientific Reports
  2025): CoT length informative only where accuracy is already high; lexical hedges are
  stronger signals.
- SELFDOUBT (arXiv 2604.06389): reports trace length as inconsistent across models;
  proposes a hedge-to-verify ratio.
- How Language Models Fail: Token-Level Signatures of Committed and Persistent Reasoning
  Failures (arXiv 2606.06635).
- Latency-Response Theory Model (arXiv 2512.07019): IRT jointly modelling accuracy and
  CoT length.
- Xue et al., An Empirical Study of Reasoning Steps in Thinking Code LLMs
  (arXiv 2511.05874): code generation (BigCodeBench); reasoning steps vs. difficulty,
  not as a confidence signal.

## Implications for the study

1. The triple overlap (output prediction × confidence × program properties) remains the
   core claim.
2. Position against Playing Psychic explicitly: real-world code with naturally varying
   properties (RE2) instead of synthetic composition depth; zero-shot signals instead of a
   trained classifier; calibration and selective-prediction metrics; a head-to-head
   comparison of effort, token probability and verifier verdict, conditioned on program
   properties to locate each signal's blind spots.
3. Anticipate "why length when trace structure is better?": closed reasoning APIs expose
   only a reasoning-token count, not the trace, so for those models length is the only
   effort signal available.
4. Two papers report trace length as inconsistent across models; the effort result needs
   several model families before it can be claimed.
5. Devic et al. leave open how trace length relates to objectively measured difficulty;
   output prediction allows difficulty to be measured from the program itself.

### Suggested additions to the related-work table

| Study | Program Output Prediction | Code Confidence Measurement & Calibration | Program Properties |
|---|---|---|---|
| Armengol-Estapé et al. (2503.05703) | ✓ | (✓) token NLL | (✓) steps ahead |
| Playing Psychic (2604.16931) | ✓ | (✓) trace-structure classifier | ✓ composition depth |
| Devic et al. (2510.10409) | – | ✓ trace length | – |

Alternatively, add a fourth column "reasoning-trace signals" to separate the effort line of
work from verbalized/verifier confidence.

## Sources

- Playing Psychic: https://arxiv.org/abs/2604.16931
- Thinking Hard, Not Smart: https://arxiv.org/abs/2608.07968
- What I cannot execute, I do not understand: https://arxiv.org/abs/2503.05703
- Open the Oyster: https://arxiv.org/abs/2511.02197
- Multicalibration for LLM-based Code Generation: https://arxiv.org/abs/2512.08810
- Trace Length is a Simple Uncertainty Signal in Reasoning Models: https://arxiv.org/abs/2510.10409
- Revisiting the Test-Time Scaling of o1-like Models: https://arxiv.org/abs/2502.12215
- Between Underthinking and Overthinking: https://arxiv.org/abs/2505.00127
- Reasoning Models Better Express Their Confidence: https://arxiv.org/abs/2505.14489
- Lexical hints of accuracy in LLM reasoning chains: https://arxiv.org/abs/2508.15842
- SELFDOUBT: https://arxiv.org/abs/2604.06389
- How Language Models Fail: https://arxiv.org/abs/2606.06635
- Latency-Response Theory Model: https://arxiv.org/abs/2512.07019
- An Empirical Study of Reasoning Steps in Thinking Code LLMs: https://arxiv.org/abs/2511.05874
- Code-MUE: https://arxiv.org/abs/2607.12273
- On the Robustness of LLMs' Internal Representation of Code Correctness: https://arxiv.org/abs/2608.08266
- Ensemble-Based Uncertainty Estimation for Code Correctness Estimation: https://arxiv.org/abs/2603.27098
- Task Abstention for LLMs in Code Generation: https://arxiv.org/abs/2605.17029
- Latent Programming Horizons in Coding Agents: https://arxiv.org/abs/2607.05188
- Semantic Scholar Graph API: https://api.semanticscholar.org/api-docs/graph
