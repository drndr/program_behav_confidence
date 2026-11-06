# Qwen3.8-27B server bundle

Runs the study's prompts through Qwen3.8-27B (bf16, Hugging Face Transformers, one 80 GB GPU)
at low reasoning, then has the same model judge its own answers (self-verification). Build the
bundle in the repository with

```sh
python scripts/build_server_bundle.py   # -> _local/server_bundles/qwen38_27b_server.zip
```

The bundle holds the notebook, the prompts, the datasets and the repository code the notebook
reuses (the runner's prompt checks, the verification prompts, answer parsing), all at their
repository paths. `bundle_manifest.json` records the commit and every file's hash; the notebook
checks them first.

## On the server

1. Upload and unzip the bundle, and open `qwen38_27b_transformers.ipynb` from its folder.
2. Run the first code cell once from the working cluster AI kernel. It installs a private
   package overlay next to the notebook and registers the kernel `Python (Qwen3.8 Overlay)`,
   without changing the cluster installation or PyTorch. Refresh JupyterLab, select that
   kernel, restart it and run from the top.
3. The checks stop on failure: bundle hashes, GPU, prompts, model placement, prompt lengths,
   token bytes, thinking control, and a one-token run on the longest prompt.
4. The pilot runs 10 prompts per dataset, spread over each file, and their self-verification,
   then prints a report and the projected hours. If something looks wrong, stop and send back
   the results zip (its cell works at any point).
5. Set `CONFIRM_FULL_RUN = True` to run all predictions (RE2 first), then the
   self-verification.
6. Make the results zip and release the GPU.

Every run resumes: after an interruption or a session time limit, rerun from the top and saved
items are skipped. A failed request (e.g. out of GPU memory) is saved as an error and retried
only with `RETRY_ERRORS = True`. Truncated answers are kept, as in the runner.

## Supplementary run: the ten added RE2 items

RE2 now keeps every sample except those taken from CRUXEval, which adds ten items to the main
run. The notebook's `ADDITIONS = True` (the default) runs only those ten
(`data/prompts/re2_additions/`) as `qwen3.8_27b_low_reasoning_re2_additions`, with its
self-verification, and skips the pilot. Run from the top as above; the results zip is
`qwen3.8_27b_low_reasoning_re2_additions_results.zip`. Set `ADDITIONS = False` for the full
study prompts.

## Back in the repository

Unzip `qwen3.8_27b_low_reasoning_results.zip` at the repository root. The `data/` files land
where the runner would have written them; `_local/qwen38_27b/` (ignored by git) holds the pilot,
the log-probability of every generated token, and the session log. Then

```sh
python scripts/evaluate_predictions.py data/predictions/qwen3.8_27b_low_reasoning.jsonl
python scripts/evaluate_verifications.py data/verifications/qwen3.8_27b_low_reasoning_self_verification.jsonl
```

and add the run to `CONDITIONS` in `src/runs.py` before `python scripts/analyze.py`.

## Differences from the GPT-6 Luna runs

- Sampling is set explicitly: temperature 1.0, top-p 0.95, top-k 20, seed 0 per item.
- Low reasoning is a sentence Qwen's chat template adds to the prompt; there is no thinking
  budget beyond the 16,384 output tokens.
- Log-probabilities come from the model's own distribution, before sampling filters, and
  exist with reasoning on, so the answer log-probability is available at low reasoning.
- Reasoning tokens count the generated tokens up to and including `</think>`.
