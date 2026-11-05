"""The paper figure of the TTS evaluation, from the distributions run (no recomputation).

Writes ``<output_dir>/attribute_ranges.{pdf,png}`` (and copies the PDF into
``paper_figures_dir`` when set): per condition, the median, 25–75 % and 5–95 % of each
``ranges`` row; effort and emotion side by side.

Usage
-----
    uv run --extra cpu launch_experiment --script scripts/make_figures.py \\
        --config configs/figures.yaml
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from experiment_launcher import parse_args
from speech_eval import figures as base

from tts_evaluation import figures

SYSTEMS = list(figures.SOURCES)
EXPERIMENTS = {"effort": "Effort", "emotion": "Emotion"}


def save(fig, out: Path, name: str, paper_dir: Path | None) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"{name}.{suffix}", bbox_inches="tight", facecolor=base.SURFACE,
                    dpi=300)
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(out / f"{name}.pdf", paper_dir / f"{name}.pdf")


def condition_order(summary: pd.DataFrame, config: dict) -> dict[str, dict[str, str]]:
    """``{experiment: {condition: tick label}}`` in display order.

    The experiments in ``sort_by_natural_leq`` are ordered by increasing median Leq of
    natural speech, pooled over sex; the others keep the order of ``conditions``.
    """
    order = {}
    for experiment, names in config["conditions"].items():
        if experiment in config["sort_by_natural_leq"]:
            part = summary[(summary["experiment"] == experiment)
                           & (summary["attribute"] == "intensity") & (summary["sex"] == "all")]
            names = {c: names[c] for c in figures.order_by_median(part, list(names))}
        order[experiment] = names
    return order


def attribute_ranges(plt, summary: pd.DataFrame, rows: list[dict],
                     conditions: dict[str, dict[str, str]]):
    """Rows = ``ranges`` entries ({attribute, sex, label, y_group?}), columns = experiments."""
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.2, 1.6 * len(rows) + 0.5),
                             gridspec_kw={"width_ratios": [len(conditions["effort"]),
                                                           len(conditions["emotion"])]},
                             sharey="row", squeeze=False)
    for r, row in enumerate(rows):
        for c, experiment in enumerate(EXPERIMENTS):
            part = summary[(summary["experiment"] == experiment)
                           & (summary["attribute"] == row["attribute"])
                           & (summary["sex"] == row["sex"])]
            names = conditions[experiment]
            ax = axes[r, c]
            figures.range_bars(ax, part, list(names), SYSTEMS, tick_labels=list(names.values()))
            if r < len(rows) - 1:
                ax.tick_params(labelbottom=False)
            figures.label_axes(ax, ylabel=row["label"] if c == 0 else None,
                               title=EXPERIMENTS[experiment] if r == 0 else None)
    # Rows with the same ``y_group`` share one y range, so their values compare directly.
    groups = {}
    for r, row in enumerate(rows):
        if row.get("y_group"):
            groups.setdefault(row["y_group"], []).extend(axes[r])
    for group in groups.values():
        lows, highs = zip(*(ax.get_ylim() for ax in group))
        for ax in group:
            ax.set_ylim(min(lows), max(highs))
    fig.legend(handles=figures.legend_handles(SYSTEMS), loc="lower center", ncol=4,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


@parse_args
def main(config: dict, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paper_dir = Path(config["paper_figures_dir"]) if config.get("paper_figures_dir") else None
    summary = pd.read_csv(Path(config["distributions"]) / "distributions.csv")
    conditions = condition_order(summary, config)
    print("condition order:", {e: list(n) for e, n in conditions.items()})
    save(attribute_ranges(plt, summary, config["ranges"], conditions), out, "attribute_ranges",
         paper_dir)
    plt.close("all")
    print(f"Figures in {out}" + (f", PDFs copied to {paper_dir}" if paper_dir else ""))


if __name__ == "__main__":
    main()
