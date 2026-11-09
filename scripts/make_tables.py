"""Write the paper's data tables as LaTeX (no model calls): python scripts/make_tables.py

  audit.tex       Table III: audit labels of the RE2 examples that no condition with reasoning
                  answered correctly, per source
  properties.tex  Table VI: median program properties per dataset, RE2 on its solvable examples
  accuracy.tex    Table VII: accuracy per condition on CRUXEval and on RE2's solvable examples
  within.tex      Table VIII: AUROC within source and code-size third on RE2's solvable examples,
                  marking signals reliably above code size, and the share of incorrect answers
                  that self-verification approves
  triage.tex      Table IX: errors caught when checking the 20% of answers each signal ranks as
                  most likely to be wrong

Reads data/analysis/ (written by scripts/analyze.py, scripts/paired_differences.py and
scripts/error_triage.py), data/audit/re2_audit.csv and data/program_properties/, and writes one
file per table to data/tables/ for \\input in the paper (needs booktabs and siunitx).
The paper's terms live here, not in the data: RE2's library code is called repository code.
"""

import argparse
import csv
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import re2_source

MODELS = {"GPT-6 Luna": "Luna", "DeepSeek V4.1 Flash": "DeepSeek", "Qwen 3.8 27B": "Qwen", "Claude Haiku 5.5": "Haiku"}
SOLVABLE = "RE2 solvable"
# RE2 sources as the paper groups them.
SOURCES = {"Real": "Projects", "Swebench": "SWE-bench"}
AUDIT_LABELS = [("input_lost", "Input lost"), ("reference_wrong", "Reference misrecorded"),
                ("environment", "Environment-dependent")]
PROPERTY_ROWS = [("Size", [("Code tokens", "code_token_length"), ("Input tokens", "input_token_length"),
                           ("Executable lines", "executable_lines")]),
                 ("Complexity", [("Cyclomatic complexity", "cyclomatic_complexity"),
                                 (r"\quad target only", "target_cyclomatic_complexity"),
                                 ("Halstead difficulty", "halstead_difficulty"),
                                 (r"\quad target only", "target_halstead_difficulty"),
                                 ("Information flow (target)", "information_flow")]),
                 ("Execution", [("Trace length", None)])]
WITHIN_PROPERTIES = ["code tokens (baseline)", "complexity (baseline)", "Halstead difficulty (baseline)",
                     "information flow (baseline)"]
WITHIN_SIGNALS = ["answer logprob", "reasoning tokens", "stated confidence"]
SIZE = "code tokens (baseline)"
TRIAGE_COLUMNS = [("Size", SIZE), ("Reasoning", "reasoning tokens"), ("Stated", "stated confidence"),
                  ("Log-prob.", "answer logprob")]
TRIAGE_BUDGET = 0.2


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def conditions(rows):
    """(model, reasoning) pairs in the order the analysis wrote them."""
    ordered = []
    for row in rows:
        if (row["model"], row["reasoning"]) not in ordered:
            ordered.append((row["model"], row["reasoning"]))
    return ordered


def condition_name(model, reasoning):
    return f"{MODELS.get(model, model)}, {reasoning}"


def bold_if(text, bold):
    return rf"\textbf{{{text}}}" if bold else text


def audit_table(audit, re2_ids):
    """Table III: audit labels per source, with the solvable split and every RE2 example."""
    def column(item_id):
        return SOURCES.get(re2_source(item_id), "Exercises")

    columns = ["Projects", "SWE-bench", "Exercises"]
    count = lambda rows: [sum(column(r["id"]) == c for r in rows) for c in columns]
    unsolvable = [r for r in audit if r["solvable"] == "False"]
    body = [(name, count([r for r in unsolvable if r["category"] == key])) for key, name in AUDIT_LABELS]
    split = [("Not solvable", count(unsolvable)), ("Solvable", count([r for r in audit if r["solvable"] == "True"]))]
    totals = [("Inspected", count(audit)), ("All examples", [sum(column(i) == c for i in re2_ids) for c in columns])]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{Audit results on the {len(audit)} RE2 examples that no condition with reasoning answered "
             r"correctly.}", r"\label{tab:audit}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Label & Projects & SWE-bench & Exercises & Total \\", r"\midrule"]
    for block in (body, split, totals):
        if block is not body:
            lines.append(r"\midrule")
        lines += [f"{name} & " + " & ".join(map(str, values)) + rf" & {sum(values)} \\" for name, values in block]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def properties_table(cruxeval, re2, traces, solvable_ids):
    """Table VI: median properties of CRUXEval and of RE2's solvable exercises and repository functions."""
    repository = lambda r: re2_source(r["id"]) in SOURCES
    groups = [cruxeval, [r for r in re2 if r["id"] in solvable_ids and not repository(r)],
              [r for r in re2 if r["id"] in solvable_ids and repository(r)]]

    def median(rows, field):
        if field is None:
            values = [traces[r["dataset"]][r["id"]] for r in rows if r["id"] in traces[r["dataset"]]]
        else:
            values = [float(r[field]) for r in rows if r[field] not in ("", None)]
        return f"{statistics.median(values):.0f}" if values else "--"

    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{Median program properties per dataset, on the {len(groups[1]) + len(groups[2])} solvable RE2 "
             r"examples.}", r"\label{tab:dataset-statistics}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{RE2} \\", r"\cmidrule(l){3-4}", r" & CRUXEval & Exercises & Repository \\",
             r"\midrule", "Examples & " + " & ".join(str(len(g)) for g in groups) + r" \\"]
    for heading, rows in PROPERTY_ROWS:
        lines += [r"\addlinespace", rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{{heading}}}}} \\"]
        lines += [f"{label} & " + " & ".join(median(g, field) for g in groups) + r" \\" for label, field in rows]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def accuracy_table(accuracy, mode="normalized"):
    """Table VII: accuracy per condition on CRUXEval and RE2's solvable examples, best per column in bold."""
    rows = [r for r in accuracy if r["mode"] == mode]
    counts = {(r["model"], r["reasoning"], r["group"]): (int(r["correct"]), int(r["items"])) for r in rows}
    order = conditions(rows)
    for model, reasoning in order:
        # Solvable exercises: the solvable RE2 examples that are not repository code.
        (right, items), (right_repo, items_repo) = (counts[(model, reasoning, SOLVABLE)],
                                                    counts[(model, reasoning, "RE2 library code solvable")])
        counts[(model, reasoning, "RE2 exercises solvable")] = (right - right_repo, items - items_repo)
    columns = ["CRUXEval", "RE2 exercises solvable", "RE2 library code solvable", SOLVABLE]
    sizes = [counts[(*order[0], group)][1] for group in columns]
    values = {(c, g): f"{100 * counts[(*c, g)][0] / counts[(*c, g)][1]:.1f}" for c in order for g in columns}
    best = {g: max(float(values[(c, g)]) for c in order) for g in columns}
    name_width = max(len(MODELS.get(model, model)) for model, _ in order)
    effort_width = max(len(reasoning) for _, reasoning in order)
    lines = [r"\begin{table}[h]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             rf"\caption{{Accuracy (\%) per condition on CRUXEval and on the {sizes[3]} solvable RE2 examples "
             r"(Section~\ref{sec:audit}). Best result per column in bold.}", r"\label{tab:accuracy}",
             r"\begin{tabular}{@{}llr|rrr@{}}", r"\toprule", r" & & & \multicolumn{3}{c}{RE2} \\", r"\cmidrule(l){4-6}",
             r"Model & Effort & CRUXEval & Exer. & Repo. & All \\", " & & " + " & ".join(f"({n})" for n in sizes) + r" \\",
             r"\midrule"]
    previous = None
    for index, (model, reasoning) in enumerate(order):
        # Models with several efforts get a block of their own, closed by a rule.
        if previous is not None and model != previous and sum(m == previous for m, _ in order) > 1:
            lines.append(r"\midrule")
        name = MODELS.get(model, model) if model != previous else ""
        row = " & ".join(bold_if(values[((model, reasoning), g)], float(values[((model, reasoning), g)]) == best[g])
                         for g in columns)
        lines.append(f"{name:<{name_width}} & {reasoning:<{effort_width}} & {row} \\\\")
        previous = model
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def auroc_cell(row, dagger=False):
    value = float(row["within_source_and_size_auroc"])
    interval = ", ".join(f"{float(row[k]):.2f}".lstrip("0") for k in ("within_source_and_size_low",
                                                                         "within_source_and_size_high"))
    return value, f"{value:.2f}" + (r"$^\dagger$" if dagger else ""), f"[{interval}]"


def within_table(properties_common, signals, paired, verification):
    """Table VIII: AUROC within source and size; properties on their shared answers, signals on their own."""
    properties = {(r["model"], r["reasoning"], r["signal"]): r for r in properties_common
                  if r["comparison"] == "static" and r["group"] == SOLVABLE}
    own = {(r["model"], r["reasoning"], r["signal"]): r for r in signals
           if r["group"] == SOLVABLE and r["within_source_and_size_auroc"]}
    above = {(r["model"], r["reasoning"], r["signal"]) for r in paired
             if r["baseline"] == SIZE and r["excludes_zero"] == "True" and float(r["difference"]) > 0}
    approved = {(r["model"], r["reasoning"]): int(r["approved_wrong"]) / (int(r["approved_wrong"]) + int(r["rejected_wrong"]))
                for r in verification if r["dataset"] == "re2" and r["subset"] == "solvable"}
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{AUROC within source and code-size third on the 332 solvable RE2 examples, with "
             r"\qty{95}{\percent} intervals. Bold: highest value in each row. Program properties use their shared "
             r"subset. Confidence signals use separate subsets based on availability. $\dagger$: reliably above code "
             r"size on the same answers, with a \qty{95}{\percent} interval of the difference that rules out equal "
             r"AUROCs. Wrong approved: share of incorrect answers judged correct.}", r"\label{tab:within}",
             r"\begin{tabular}{@{}lcccccccr@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{Program properties} & \multicolumn{3}{c}{Confidence signals} & \\",
             r"\cmidrule(lr){2-5} \cmidrule(lr){6-8}",
             r" & Code & Cyclomatic & Halstead & Information & Answer & Reasoning & Stated & Wrong \\",
             r"Condition & size & complexity & difficulty & flow & log-prob. & tokens & confidence & approved \\",
             r"\midrule"]
    for model, reasoning in conditions(properties_common):
        cells = [auroc_cell(properties[(model, reasoning, name)]) for name in WITHIN_PROPERTIES]
        for name in WITHIN_SIGNALS:
            row = own.get((model, reasoning, name))
            cells.append(auroc_cell(row, (model, reasoning, name) in above) if row else None)
        best = max(f"{c[0]:.2f}" for c in cells if c)
        text = ["--" if c is None else f"{bold_if(c[1], f'{c[0]:.2f}' == best)} {c[2]}" for c in cells]
        share = approved.get((model, reasoning))
        text.append(rf"\qty{{{100 * share:.0f}}}{{\percent}}" if share is not None else "--")
        lines.append(f"{condition_name(model, reasoning)} & " + " & ".join(text) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])


def triage_table(triage, budget=TRIAGE_BUDGET):
    """Table IX: share of errors caught (%) at the budget, per condition and signal."""
    caught = {(r["model"], r["reasoning"], r["signal"]): float(r["errors_caught"]) for r in triage
              if float(r["budget"]) == budget and r["errors_caught"]}
    percent = f"{100 * budget:.0f}\\%"
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{Errors caught (\%) when checking the {percent} of answers each signal ranks as most likely to "
             rf"be wrong, on solvable RE2 answers that have all signals. Random checking catches {percent}.}}",
             r"\label{tab:triage}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             "Condition & " + " & ".join(name for name, _ in TRIAGE_COLUMNS) + r" \\", r"\midrule"]
    for model, reasoning in conditions(triage):
        values = [caught.get((model, reasoning, signal)) for _, signal in TRIAGE_COLUMNS]
        lines.append(f"{condition_name(model, reasoning)} & "
                     + " & ".join("--" if v is None else f"{100 * v:.0f}" for v in values) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def traces(properties_dir):
    """Trace lengths of the examples whose recorded output the execution reproduced, per dataset."""
    out = {}
    for name in ("cruxeval", "re2"):
        rows = read(properties_dir / f"{name}_execution.csv")
        out[name] = {r["id"]: int(r["trace_length"]) for r in rows if r["output_matches"] == "True" and r["trace_length"]}
    return out


def make_tables(data_dir=ROOT / "data", out_dir=ROOT / "data" / "tables"):
    analysis, properties_dir = data_dir / "analysis", data_dir / "program_properties"
    audit = read(data_dir / "audit" / "re2_audit.csv")
    re2 = read(properties_dir / "re2_properties.csv")
    solvable = {r["id"] for r in re2} - {r["id"] for r in audit if r["solvable"] == "False"}
    tables = {"audit": audit_table(audit, [r["id"] for r in re2]),
              "properties": properties_table(read(properties_dir / "cruxeval_properties.csv"), re2, traces(properties_dir),
                                             solvable),
              "accuracy": accuracy_table(read(analysis / "accuracy.csv")),
              "within": within_table(read(analysis / "properties_common.csv"), read(analysis / "signals.csv"),
                                     read(analysis / "paired.csv"), read(analysis / "verification.csv")),
              "triage": triage_table(read(analysis / "triage.csv"))}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in tables.items():
        (out_dir / f"{name}.tex").write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {out_dir / f'{name}.tex'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "tables")
    args = parser.parse_args()
    make_tables(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
