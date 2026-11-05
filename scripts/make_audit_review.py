"""Write a blind review page for checking the RE2 audit: python scripts/make_audit_review.py [--size N]

Draws a random sample of the audited items (data/audit/re2_audit.csv), stratified by the
audit's verdict (input lost, reference misrecorded, environment-dependent, solvable), and
writes data/audit/review/review.html, a self-contained page that shows each sampled item's
task, code, input, reference and the study conditions' answers, but not the audit's labels.
A reviewer picks one of these verdicts for each item, gives a reason, and exports a CSV; compare
the exports with scripts/audit_agreement.py. The sampled ids go to data/audit/review/sample.txt.
"""

import argparse
import csv
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datasets import DATASET_SOURCES, load_examples, re2_source
from prompt_template import RE2_INPUT_NOTE, task_sentence
from runs import AUDIT, CONDITIONS
from scoring import parse_prediction

OUT = ROOT / "data/audit/review"
SEED = 2026
SIZE = 40


def verdict(row):
    """The audit's verdict on an item: "solvable", or the category of why it is not."""
    return "solvable" if row["solvable"] == "True" else row["category"]


def stratified_sample(rows, size, seed=SEED):
    """Ids of `size` audited items, each verdict sampled in proportion to its share (largest remainders)."""
    strata = {}
    for row in rows:
        strata.setdefault(verdict(row), []).append(row["id"])
    exact = {name: size * len(ids) / len(rows) for name, ids in strata.items()}
    counts = {name: int(share) for name, share in exact.items()}
    for name in sorted(exact, key=lambda name: exact[name] - counts[name], reverse=True)[:size - sum(counts.values())]:
        counts[name] += 1
    rng = random.Random(seed)
    return sorted(i for name in sorted(strata) for i in rng.sample(sorted(strata[name]), counts[name]))


def answers(ids, data_dir):
    """Per item, the answers of the study conditions: each distinct answer with the conditions that gave
    it and whether it was scored correct (normalized scoring)."""
    given = {i: {} for i in ids}
    for condition in CONDITIONS:
        label = f"{condition['model']}, reasoning {condition['reasoning']}"
        for run in condition["runs"]:
            with (data_dir / "evaluations" / f"{run}_scores.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["dataset"] != "re2" or row["id"] not in given or row["mode"] != "normalized":
                        continue
                    if row["status"].startswith("request_"):
                        text = "(no answer: the response was cut off or the request failed)"
                    else:
                        try:
                            text = parse_prediction(row["prediction"] or "")
                        except ValueError:
                            text = "(no single <prediction> tag in the response)"
                    given[row["id"]][label] = (text, row["correct"])
    result = {}
    for item_id, by_condition in given.items():
        groups = {}
        for label, (text, correct) in by_condition.items():
            groups.setdefault((text, correct), []).append(label)
        result[item_id] = [{"answer": text, "correct": correct, "conditions": labels}
                           for (text, correct), labels in groups.items()]
    return result


def review_items(ids, data_dir=ROOT / "data"):
    examples = {row["id"]: row for row in load_examples(data_dir / DATASET_SOURCES["re2"], "re2")}
    given = answers(ids, data_dir)
    return [{"id": i, "source": re2_source(i), "task": task_sentence(examples[i]), "code": examples[i]["code"],
             "input": examples[i]["input"], "reference": examples[i]["output"], "answers": given[i]} for i in ids]


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RE2 audit review</title>
<style>
  body { margin: 0; font: 14px/1.45 system-ui, sans-serif; color: #1d1d1f; background: #fff;
         display: grid; grid-template-columns: 250px 1fr; height: 100vh; }
  nav { overflow: auto; border-right: 1px solid #ccc; padding: 10px; background: #fafafa; }
  nav input { width: 100%; box-sizing: border-box; margin: 4px 0 8px; }
  nav .item { display: block; width: 100%; text-align: left; margin: 2px 0; padding: 3px 5px; cursor: pointer;
              font: 12px ui-monospace, monospace; border: 1px solid #ddd; background: #fff;
              overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  nav .item.done { background: #e3f2e3; }
  nav .item.current { outline: 2px solid #3366cc; }
  main { overflow: auto; padding: 14px 22px 40px; }
  h2 { font-size: 16px; word-break: break-all; margin: 6px 0; }
  .label { font-weight: 600; margin-top: 12px; }
  pre { background: #f4f4f4; padding: 8px; margin: 4px 0; max-height: 420px; overflow: auto;
        white-space: pre-wrap; word-break: break-word; font: 12px/1.4 ui-monospace, monospace; }
  table { border-collapse: collapse; width: 100%; }
  td { border-top: 1px solid #ddd; padding: 4px 6px; vertical-align: top; }
  td.mark { width: 70px; }
  fieldset { margin-top: 16px; border: 1px solid #bbb; }
  textarea { width: 100%; box-sizing: border-box; min-height: 60px; }
  .row { margin: 6px 0; }
  .muted { color: #666; }
  details { margin-bottom: 10px; }
  @media (max-width: 800px) {
    body { display: block; height: auto; }
    nav { max-height: 35vh; border-right: none; border-bottom: 1px solid #ccc; }
  }
</style>
</head>
<body>
<nav>
  <label>Your name<input id="rater" placeholder="e.g. reviewer-a"></label>
  <div id="progress" class="muted"></div>
  <button id="export">Export CSV</button>
  <div id="list"></div>
</nav>
<main>
  <details>
    <summary><b>Instructions</b></summary>
    <p>Each item is an RE2 example that no study condition with reasoning answered correctly. Judge from the
    code and input as given, without running anything, whether the reference output can be derived from them,
    and pick one verdict. Do not open <code>data/audit/re2_audit.csv</code> or the other reviewer's labels
    before you finish.</p>
    <ul>
      <li><b>Solvable</b>: the reference can be derived. In the reason, say why the models missed it, using their
      answers below, e.g. the right value in a form the scoring rejects, a long exact computation, a library's
      output format, or errors in following the code.</li>
      <li><b>Input lost</b>: the input lacks state the output depends on, e.g. objects recorded as null or as
      partial attribute dictionaries, arrays abbreviated with an ellipsis, fitted models or test-local classes
      not given.</li>
      <li><b>Reference misrecorded</b>: the reference is not what the code returns, e.g. a returned object
      recorded as an attribute dictionary instead of its representation, or an exception or empty list recorded
      as null.</li>
      <li><b>Environment-dependent</b>: the output depends on the execution, e.g. memory addresses in
      representations, randomness, hash randomization, style or font settings.</li>
    </ul>
    <p class="muted" id="note"></p>
    <p class="muted">Labels are saved in this browser as you go. Export a CSV when done and send it back.</p>
  </details>
  <div id="item"></div>
</main>
<script>
const DATA = __DATA__;
const VERDICTS = [["solvable", "Solvable"], ["input_lost", "Input lost"], ["reference_wrong", "Reference misrecorded"],
                  ["environment", "Environment-dependent"]];
const KEY = "re2-audit-review-verdicts-" + DATA.seed;
const $ = id => document.getElementById(id);
let state = {rater: "", labels: {}}, current = 0;
try { state = JSON.parse(localStorage.getItem(KEY)) || state; } catch (error) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (error) {} };
const done = id => { const label = state.labels[id]; return label && label.verdict; };

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function renderList() {
  $("list").replaceChildren(...DATA.items.map((item, index) => {
    const button = element("button", `${index + 1}. ${item.id}`, "item" + (done(item.id) ? " done" : "") +
                           (index === current ? " current" : ""));
    button.title = item.id;
    button.onclick = () => { current = index; render(); };
    return button;
  }));
  $("progress").textContent = `${DATA.items.filter(item => done(item.id)).length} of ${DATA.items.length} labelled`;
}

function section(label, text) {
  return [element("div", label, "label"), element("pre", text)];
}

function render() {
  const item = DATA.items[current];
  const label = state.labels[item.id] || (state.labels[item.id] = {verdict: "", reason: ""});
  const view = $("item");
  view.replaceChildren(element("h2", `${current + 1} / ${DATA.items.length}  ${item.id}`),
                       element("div", `${item.source} · ${item.task}`, "muted"),
                       ...section("Code", item.code), ...section("Input", item.input),
                       ...section("Reference output", item.reference));
  view.append(element("div", "Answers of the study conditions", "label"));
  const table = element("table");
  for (const answer of item.answers) {
    const row = table.insertRow();
    row.insertCell().append(element("pre", answer.answer));
    const mark = row.insertCell();
    mark.className = "mark";
    mark.textContent = answer.correct === true ? "correct" : answer.correct === false ? "incorrect" : "unscored";
    row.insertCell().textContent = answer.conditions.join("; ");
  }
  view.append(table);

  const form = element("fieldset");
  form.append(element("legend", "Your judgment"));
  const verdict = element("div", undefined, "row");
  verdict.append("Can the reference be derived from the code and input as given? ");
  for (const [value, text] of VERDICTS) {
    const input = element("input");
    input.type = "radio";
    input.name = "verdict";
    input.checked = label.verdict === value;
    input.onchange = () => { label.verdict = value; save(); renderList(); };
    const wrap = element("label");
    wrap.append(input, " " + text + " ");
    verdict.append(wrap);
  }
  form.append(verdict);
  const reason = element("textarea");
  reason.placeholder = "Reason in one line: which input field lacks what, which line differs, or why the models missed it";
  reason.value = label.reason;
  reason.oninput = () => { label.reason = reason.value; save(); };
  form.append(element("div", "Reason", "row"), reason);
  const buttons = element("div", undefined, "row");
  const previous = element("button", "← Previous");
  previous.disabled = current === 0;
  previous.onclick = () => { current--; render(); };
  const next = element("button", "Next →");
  next.disabled = current === DATA.items.length - 1;
  next.onclick = () => { current++; render(); };
  buttons.append(previous, " ", next);
  form.append(buttons);
  view.append(form);
  view.scrollTop = 0;
  renderList();
}

function csvField(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}

$("rater").value = state.rater;
$("rater").oninput = () => { state.rater = $("rater").value.trim(); save(); };
$("note").textContent = "How RE2 records inputs (as told to the models): " + DATA.note;
$("export").onclick = () => {
  if (!state.rater) { alert("Enter your name first."); $("rater").focus(); return; }
  const missing = DATA.items.filter(item => !done(item.id)).length;
  if (missing && !confirm(`${missing} items are not labelled yet. Export anyway?`)) return;
  const lines = [["id", "rater", "verdict", "reason"].join(",")];
  for (const item of DATA.items) {
    const label = state.labels[item.id] || {};
    lines.push([item.id, state.rater, label.verdict, label.reason].map(csvField).join(","));
  }
  const link = element("a");
  link.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], {type: "text/csv"}));
  link.download = `re2_audit_review_${state.rater.replace(/[^A-Za-z0-9_-]+/g, "_")}.csv`;
  link.click();
};
render();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=SIZE, help=f"items to sample (default {SIZE})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"sampling seed (default {SEED})")
    args = parser.parse_args()
    with (ROOT / "data" / AUDIT).open(encoding="utf-8", newline="") as stream:
        audited = list(csv.DictReader(stream))
    ids = stratified_sample(audited, args.size, args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample.txt").write_text("".join(i + "\n" for i in ids), encoding="utf-8")
    data = json.dumps({"seed": args.seed, "note": RE2_INPUT_NOTE, "items": review_items(ids)}, ensure_ascii=False)
    (OUT / "review.html").write_text(PAGE.replace("__DATA__", data.replace("</", "<\\/")), encoding="utf-8")
    print(f"{len(ids)} of {len(audited)} audited items -> {(OUT / 'review.html').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
