# Re-execution check of the reviewed audit sample

The 40 examples of the audit's review sample (`../review/sample.txt`) were re-executed to check
their labels against the code's actual behavior:

- **SWE-bench functions (25):** in the official SWE-bench evaluation image of each instance,
  with RE2's own patch (fix, tests, and instrumentation) applied and its tests run. RE2's
  instrumentation was kept, and each call was also logged with its serializer's list bug fixed
  and with the real return value.
- **Project functions (13):** under the tests of library releases whose code matches the
  supplied function, logging each call in the format of RE2's project records (RE2's own
  instrumentation for these is not in its archive). Two inputs were rebuilt and run directly.
- **Exercises (2):** with RE2's instrumented ClassEval files and their tests, also logging each
  call's state before it ran.

`re2_reexecution.csv` holds one label per example, in the format of the reviews, so that

```sh
python scripts/audit_agreement.py data/audit/review/re2_audit_review_reviewer-A.csv data/audit/review/re2_audit_review_reviewer-F.csv data/audit/reexecution/re2_reexecution.csv
```

compares it with the audit and both reviews. Each row gives the label (`verdict`), a one-line
observation (`reason`), the `method`, and two further columns:

- `evidence` says what supports the label.
  - `rerun outcome` (31 examples): what happened when the code ran shows it. Calls with the
    recorded input returned different outputs (input lost), the real return value differs from
    its recorded form (reference misrecorded), the output changed between runs or machines
    (environment-dependent), or the recorded input reproduced the reference (solvable).
  - `verified omission` (9 examples): the rerun is consistent with the label but cannot show it
    on its own, because the calls found return the reference or none matches the recorded input
    exactly. Instead, the code shows that the output depends on something the recorded input
    leaves out: a class or command defined only in the tests, a mock's configured return value,
    rows hidden by numpy's abbreviation, or the NumPy version (one example uses an alias that
    newer versions removed).
- `re2_records_matched` checks the setup, not the label: of RE2's records for the function, how
  many the rerun of RE2's own instrumentation produced exactly (for example, "46 of 48"). It is
  empty for project functions, whose instrumentation is not in RE2's archive.

`evidence/<id>.json` holds, per example, RE2's record and the calls whose input matches it,
grouped by their real output. Values longer than 1,500 characters are cut.
