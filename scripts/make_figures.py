"""Draw the paper's figures from the analysis tables (no model calls): python scripts/make_figures.py

  figure1_properties  Fig. 1: AUROC of the program properties on RE2's solvable examples, compared on
                      the same answers
  figure2_signals     Fig. 2: AUROC of the confidence signals on RE2's solvable examples, each on the
                      answers that have it

Both show GPT-6 Luna without reasoning and every model at low effort, on all examples (hollow
markers) and only on pairs of answers from the same source and code-size third (filled markers).

Reads the CSVs written by scripts/analyze.py and writes each figure as PDF (for the paper) and
PNG (for a quick look) to data/figures/. Normalized scoring throughout, as in the analysis.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import transforms
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOLVABLE = "RE2 solvable"  # the RE2 items the audit found solvable as given
# Without reasoning (Luna) and at low effort (every model); Luna's medium and high efforts stay within
# 0.04 of low.
CONDITION_STYLES = [("GPT-6 Luna", "none", "Luna, none", "o", "#A0A0A0"),
                    ("GPT-6 Luna", "low", "Luna, low", "o", "#303030"),
                    ("DeepSeek V4.1 Flash", "low", "DeepSeek, low", "s", "#D55E00"),
                    ("Qwen 3.8 27B", "low", "Qwen, low", "^", "#009E73"),
                    ("Claude Haiku 5.5", "low", "Haiku, low", "D", "#7A5195")]
# (signal, row label, heading of its group); the figures separate the groups.
KEY_PROPERTIES = [("code tokens (baseline)", "Code tokens", "Size"), ("input tokens (baseline)", "Input tokens", "Size"),
                  ("complexity (baseline)", "Cyclomatic complexity", "Complexity"),
                  ("Halstead difficulty (baseline)", "Halstead difficulty", "Complexity"),
                  ("information flow (baseline)", "Information flow", "Complexity")]
KEY_SIGNALS = [("answer logprob", "Answer log-probability", "Implicit"),
               ("reasoning tokens", "Reasoning tokens", "Behavioral"),
               ("generated tokens", "Generated tokens", "Behavioral"),
               ("verdict", "Self-verdict", "Explicit"), ("verdict P(correct)", "Verdict probability", "Explicit"),
               ("stated confidence", "Stated confidence", "Explicit")]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    return float(value) if value not in ("", None) else None


def save(figure, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(figure)


def chance(axis, vertical=True):
    (axis.axvline if vertical else axis.axhline)(0.5, color="#BBBBBB", linestyle="--", linewidth=0.8, zorder=0)


def dumbbells(axis, rows, quantities, gap=0.7):
    """Per quantity and condition, AUROC on RE2's solvable items, on all examples (hollow) joined to that within
    source and size (filled).

    Rows are grouped under bold headings in the label column, with a gap between groups and a faint line
    between the rows of a group. Returns each row's position, top to bottom.
    """
    positions, position = [], 0.0
    for index, (_, _, group) in enumerate(quantities):
        if index and group != quantities[index - 1][2]:
            position -= gap
        positions.append(position)
        position -= 1
    labels = transforms.blended_transform_factory(axis.transAxes, axis.transData)
    for index, (top, (signal, _, group)) in enumerate(zip(positions, quantities)):
        if not index or group != quantities[index - 1][2]:
            axis.text(-0.02, top + 0.62, group, transform=labels, ha="right", va="center", fontweight="bold")
        else:
            axis.axhline(top + 0.5, color="#E2E2E2", linewidth=0.6, zorder=0)
        for lane, (model, reasoning, _, marker, colour) in enumerate(CONDITION_STYLES):
            found = next((r for r in rows if r["model"] == model and r["reasoning"] == reasoning
                          and r["group"] == SOLVABLE and r["signal"] == signal), None)
            if not found:  # e.g. log-probabilities with reasoning, reasoning tokens without
                continue
            overall, within = number(found["auroc"]), number(found["within_source_and_size_auroc"])
            y = top + 0.32 - lane * 0.64 / (len(CONDITION_STYLES) - 1)
            axis.plot([within, overall], [y, y], color=colour, linewidth=0.8)
            axis.plot(overall, y, marker, color=colour, markersize=3.2, markerfacecolor="white", markeredgewidth=0.8)
            axis.plot(within, y, marker, color=colour, markersize=3.2)
    axis.set_yticks(positions, [name for _, name, _ in quantities])
    axis.tick_params(axis="y", length=0)
    axis.set_ylim(positions[-1] - 0.55, positions[0] + 1.0)
    return positions


def condition_handles():
    return [plt.Line2D([], [], color=colour, marker=marker, markersize=4, linestyle="", label=label)
            for _, _, label, marker, colour in CONDITION_STYLES]


def figure_auroc(rows, quantities, out_dir, name):
    """One column wide: the dumbbells, the two ends named in the top right corner, the conditions below."""
    ends = [plt.Line2D([], [], color="#4D4D4D", marker="o", markersize=4, linestyle="", markerfacecolor="white",
                       label="all examples"),
            plt.Line2D([], [], color="#4D4D4D", marker="o", markersize=4, linestyle="", label="within source and size")]
    figure, axis = plt.subplots(figsize=(3.5, 2.6))
    chance(axis)
    dumbbells(axis, rows, quantities)
    axis.set_xlabel("AUROC on solvable RE2")
    axis.set_xlim(0.4, 0.95)
    bottom, top = axis.get_ylim()
    axis.set_ylim(bottom, top + 0.7)
    axis.legend(handles=ends, loc="upper right", frameon=False, fontsize=6.5, handletextpad=0.2, borderaxespad=0.0)
    figure.tight_layout()
    figure.legend(handles=condition_handles(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(figure, out_dir, name)


def make_figures(analysis_dir=ROOT / "data/analysis", out_dir=ROOT / "data/figures"):
    analysis_dir, out_dir = Path(analysis_dir), Path(out_dir)
    properties = [r for r in read(analysis_dir / "properties_common.csv") if r["comparison"] == "static"]
    figure_auroc(properties, KEY_PROPERTIES, out_dir, "figure1_properties")
    figure_auroc(read(analysis_dir / "signals.csv"), KEY_SIGNALS, out_dir, "figure2_signals")
    return sorted(p.name for p in out_dir.iterdir())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "data/analysis")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/figures")
    args = parser.parse_args()
    print("\n".join(make_figures(args.analysis_dir, args.out_dir)))
