"""Figure building blocks that ``speech_eval.figures`` does not have.

Same conventions as ``speech_eval.figures`` (its colour tokens and ``style_axes`` are reused):
recessive chrome and a legend for every multi-series panel. Four series share every panel:
the three systems take the three validated categorical slots in a fixed order, and natural
speech is drawn in grey as the reference the systems are read against, so no fourth hue is
needed.

    order_by_median     conditions in increasing order of one source's median
    range_bars          median, 25–75 % and 5–95 % of an attribute per condition and source
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from speech_eval import figures as base

NATURAL_LINE = "#77756f"
#: Display order, label and colour; natural speech first, as the reference.
SOURCES = {
    "natural": ("Natural speech", NATURAL_LINE),
    "cosyvoice3": ("CosyVoice 3", base.SERIES[0]),
    "indextts2": ("IndexTTS2", base.SERIES[1]),
    "qwen3tts": ("Qwen3-TTS", base.SERIES[2]),
}


def label_axes(ax, xlabel: str | None = None, ylabel: str | None = None,
               title: str | None = None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=8, color=base.INK_PRIMARY)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=8, color=base.INK_PRIMARY)
    if title is not None:
        ax.set_title(title, fontsize=9, color=base.INK_PRIMARY, loc="left")


def order_by_median(summary: pd.DataFrame, conditions: Sequence[str],
                    system: str = "natural") -> list[str]:
    """``conditions`` sorted by increasing ``q50`` of ``system`` in ``summary`` (one row per
    (condition, system), as ``range_bars`` takes it). Every condition must have that row."""
    medians = summary[summary["system"] == system].set_index("condition")["q50"]
    missing = [c for c in conditions if c not in medians.index]
    if missing:
        raise ValueError(f"no {system} median for {missing}")
    if medians.index.has_duplicates:
        raise ValueError(f"several {system} rows per condition: filter the summary first")
    return sorted(conditions, key=lambda c: medians[c])


def range_bars(ax, summary: pd.DataFrame, categories: Sequence[str],
               systems: Sequence[str], tick_labels: Sequence[str] | None = None,
               width: float = 0.8) -> None:
    """One mark per (category, system), dodged inside the category: a dot at ``q50``, a
    thick bar ``q25``–``q75``, a thin line ``q05``–``q95``. ``summary`` holds one row per
    (condition, system) with those columns."""
    step = width / len(systems)
    for k, system in enumerate(systems):
        _, colour = SOURCES[system]
        part = summary[summary["system"] == system].set_index("condition")
        for i, category in enumerate(categories):
            if category not in part.index:
                continue
            row = part.loc[category]
            x = i - width / 2 + step * (k + 0.5)
            ax.plot([x, x], [row["q05"], row["q95"]], color=colour, linewidth=0.8,
                    solid_capstyle="butt", zorder=2)
            ax.plot([x, x], [row["q25"], row["q75"]], color=colour, linewidth=3.0,
                    solid_capstyle="butt", zorder=3)
            ax.scatter([x], [row["q50"]], s=10, color=base.SURFACE, edgecolors=colour,
                       linewidths=1.0, zorder=4)
    ax.set_xticks(range(len(categories)), tick_labels or categories, fontsize=7)
    ax.set_xlim(-0.5, len(categories) - 0.5)
    base.style_axes(ax, grid="y")
    ax.tick_params(labelsize=7)


def legend_handles(systems: Sequence[str]) -> list:
    """Proxy handles in display order, one per source."""
    from matplotlib.lines import Line2D

    return [Line2D([], [], color=SOURCES[s][1], linewidth=2.0, label=SOURCES[s][0])
            for s in systems]
