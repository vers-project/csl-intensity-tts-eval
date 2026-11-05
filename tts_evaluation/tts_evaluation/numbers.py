"""Numbers quoted in the paper text, as LaTeX macros computed from the distributions.

Every value is the median Leq (``q50``) of one (experiment, system, condition) cell of
``distributions.csv`` (sex pooled), or derived from those medians, at one decimal:

    \\med<Exp><Sys><Cond>        median of the cell
    \\gap<Exp><Sys><Cond>        |system − natural| of that condition
    \\maxgap<Exp><Sys>           largest \\gap of the system over the conditions
    \\step<Sys><Cond>            |median − median of the previous effort level|
    \\range<Exp><Sys>            effort: very loud − soft; emotion: max − min over emotions
    \\low<Exp><Sys>, \\high<Exp><Sys>
                                 lowest 5th and highest 95th percentile over the conditions
    \\diff<Name>                 |a − b| for the named cell pairs of the config
    \\failN / \\failTotal / \\failRate<Exp><Sys><Cond>
                                 failed samples (no speech found), samples, and their
                                 percentage, from the attribute table
    \\failMaxRateOther          highest failure rate of a system cell outside ``exclude``
    \\failMaxRateNatural        highest failure rate of a natural-speech cell
    \\rmsSpread<Exp><Sys>        largest − smallest median RMS level (dBFS) over the conditions
                                 (``rms_numbers``)
    \\spearman<Exp><Sys>, \\range<Exp><Sys>Alpha, \\alphaShare<Exp>Min / Max
                                 order and range of the medians, predictor against alpha
                                 (``alpha_agreement``)
    \\corrLabelPredictor, \\corrLabelAlpha, \\rangeEffNaturalLabel
                                 the two estimates against the calibrated level of AVID, from
                                 one row per sample (``calibrated_numbers``)

Differences are taken between the ROUNDED medians, so that a difference printed next to its
two medians always matches them on the page.  Magnitudes only: the text carries the sign in
words ("louder", "softer"), and ``signed_listing`` prints the signs for checking.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

EXPERIMENTS = {"effort": "Eff", "emotion": "Emo"}
SYSTEMS = {"natural": "Natural", "cosyvoice3": "Cosy", "indextts2": "Index", "qwen3tts": "Qwen"}
NATURAL = "natural"


def macro_word(name: str) -> str:
    """``veryloud`` → ``VeryLoud``; a LaTeX macro name holds letters only."""
    word = {"veryloud": "VeryLoud"}.get(name, name[:1].upper() + name[1:])
    if not word.isalpha():
        raise ValueError(f"{name!r} does not give a letters-only macro name")
    return word


def round1(value: float) -> Decimal:
    """One decimal, half away from zero (not Python's binary-float rounding)."""
    return Decimal(repr(float(value))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def medians(summary: pd.DataFrame, attribute: str = "intensity") -> pd.DataFrame:
    """``q50`` of the pooled-sex ``attribute`` rows: index (experiment, condition), columns =
    systems."""
    part = summary[(summary["attribute"] == attribute) & (summary["sex"] == "all")]
    if part.empty:
        raise ValueError(f"no pooled {attribute} row")
    if part.duplicated(["experiment", "system", "condition"]).any():
        raise ValueError(f"several pooled {attribute} rows for one cell")
    return part.pivot_table(index=["experiment", "condition"], columns="system", values="q50")


def paper_numbers(summary: pd.DataFrame, conditions: dict[str, list[str]],
                  differences: dict[str, tuple[list[str], list[str]]]) -> dict[str, Decimal]:
    """``{macro name: value}``. ``conditions`` = display order per experiment (effort order
    defines the steps); ``differences`` = {name: ([exp, sys, cond], [exp, sys, cond])}."""
    table = medians(summary)
    rounded = {}
    for experiment, names in conditions.items():
        for system in SYSTEMS:
            for condition in names:
                value = table.loc[(experiment, condition), system]
                if pd.isna(value):
                    raise ValueError(f"no median for {experiment} / {system} / {condition}")
                rounded[experiment, system, condition] = round1(value)

    out = {}
    for (experiment, system, condition), value in rounded.items():
        stem = f"{EXPERIMENTS[experiment]}{SYSTEMS[system]}{macro_word(condition)}"
        out[f"med{stem}"] = value
        if system != NATURAL:
            out[f"gap{stem}"] = abs(value - rounded[experiment, NATURAL, condition])

    for experiment, names in conditions.items():
        for system in SYSTEMS:
            if system != NATURAL:
                stem = f"{EXPERIMENTS[experiment]}{SYSTEMS[system]}"
                out[f"maxgap{stem}"] = max(out[f"gap{stem}{macro_word(c)}"] for c in names)

    effort = conditions["effort"]
    for system in SYSTEMS:
        for lower, upper in zip(effort, effort[1:]):
            out[f"step{SYSTEMS[system]}{macro_word(upper)}"] = abs(
                rounded["effort", system, upper] - rounded["effort", system, lower])
        out[f"rangeEff{SYSTEMS[system]}"] = (rounded["effort", system, effort[-1]]
                                             - rounded["effort", system, effort[0]])
        emotion = [rounded["emotion", system, c] for c in conditions["emotion"]]
        out[f"rangeEmo{SYSTEMS[system]}"] = max(emotion) - min(emotion)

    pooled = summary[(summary["attribute"] == "intensity") & (summary["sex"] == "all")]
    for experiment, names in conditions.items():
        for system in SYSTEMS:
            cells = pooled[(pooled["experiment"] == experiment) & (pooled["system"] == system)
                           & pooled["condition"].isin(names)]
            stem = f"{EXPERIMENTS[experiment]}{SYSTEMS[system]}"
            out[f"low{stem}"] = round1(cells["q05"].min())
            out[f"high{stem}"] = round1(cells["q95"].max())

    for name, (a, b) in differences.items():
        if not name.isalpha():
            raise ValueError(f"difference name {name!r} must be letters only")
        out[f"diff{name}"] = abs(rounded[tuple(a)] - rounded[tuple(b)])
    return out


def round2(value: float) -> Decimal:
    return Decimal(repr(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def spread(values) -> Decimal:
    """Largest − smallest of the rounded values."""
    rounded = [round1(v) for v in values]
    return max(rounded) - min(rounded)


def alpha_agreement(summary: pd.DataFrame, conditions: dict[str, list[str]],
                    attribute: str) -> dict[str, Decimal]:
    """Order and range of the condition medians, predictor against alpha-ratio regression.

    Per system: Spearman ρ between the two estimates' (unrounded) medians over the conditions
    (``\\spearman<Exp><Sys>``), and the range of the alpha medians (``\\range<Exp><Sys>Alpha``,
    largest − smallest).  Per experiment: the smallest and largest share, in %, of the
    predictor's range (largest − smallest) that the alpha range keeps, over the systems
    (``\\alphaShare<Exp>Min`` / ``Max``).
    """
    predictor, alpha = medians(summary), medians(summary, attribute)
    out: dict[str, Decimal] = {}
    for experiment, names in conditions.items():
        stem, share = EXPERIMENTS[experiment], {}
        for system in SYSTEMS:
            p = predictor.loc[experiment].loc[names, system]
            a = alpha.loc[experiment].loc[names, system]
            if p.isna().any() or a.isna().any():
                raise ValueError(f"missing median for {experiment} / {system}")
            out[f"spearman{stem}{SYSTEMS[system]}"] = round2(p.corr(a, method="spearman"))
            out[f"range{stem}{SYSTEMS[system]}Alpha"] = spread(a)
            share[system] = (100 * spread(a) / spread(p)).quantize(Decimal("1"),
                                                                   rounding=ROUND_HALF_UP)
        out[f"alphaShare{stem}Min"] = min(share.values())
        out[f"alphaShare{stem}Max"] = max(share.values())
    return out


def calibrated_numbers(rows: pd.DataFrame, names: list[str], label: str, predictor: str,
                       alpha: str) -> dict[str, Decimal]:
    """On calibrated natural speech (one row per sample): Pearson r of each estimate with the
    calibrated level, over the samples (``\\corrLabelPredictor``, ``\\corrLabelAlpha``), and
    the range (largest − smallest) of the calibrated condition medians
    (``\\rangeEffNaturalLabel``).  Samples lacking any of the three values are left out."""
    rows = rows.dropna(subset=[label, predictor, alpha])
    label_medians = rows.groupby("condition")[label].median()
    if set(label_medians.index) != set(names):
        raise ValueError(f"conditions {sorted(label_medians.index)} are not {sorted(names)}")
    return {"corrLabelPredictor": round2(rows[label].corr(rows[predictor])),
            "corrLabelAlpha": round2(rows[label].corr(rows[alpha])),
            "rangeEffNaturalLabel": spread(label_medians)}


def rms_numbers(summary: pd.DataFrame, conditions: dict[str, list[str]],
                attribute: str) -> dict[str, Decimal]:
    """Spread of the median RMS level over the conditions, between rounded medians."""
    table = medians(summary, attribute)
    out = {}
    for experiment, names in conditions.items():
        for system in SYSTEMS:
            values = [round1(table.loc[(experiment, c), system]) for c in names]
            out[f"rmsSpread{EXPERIMENTS[experiment]}{SYSTEMS[system]}"] = max(values) - min(values)
    return out


def signed_listing(summary: pd.DataFrame, conditions: dict[str, list[str]]) -> pd.DataFrame:
    """System − natural and step values WITH their sign, for checking the words of the text."""
    table = medians(summary)
    rows = []
    for experiment, names in conditions.items():
        for condition in names:
            natural = round1(table.loc[(experiment, condition), NATURAL])
            for system in SYSTEMS:
                if system == NATURAL:
                    continue
                rows.append({"experiment": experiment, "condition": condition, "system": system,
                             "system_minus_natural": round1(table.loc[(experiment, condition),
                                                                      system]) - natural})
    return pd.DataFrame(rows)


def latex(numbers: dict[str, Decimal | int], source: str) -> str:
    lines = [
        "% Generated by tts_evaluation/scripts/make_paper_numbers.py;",
        f"% do not edit by hand.  Source: {source}",
    ]
    lines += [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in numbers.items()]
    return "\n".join(lines) + "\n"


def failure_counts(attributes: pd.DataFrame, queries: dict[str, str | None]) -> pd.DataFrame:
    """``failed`` and ``total`` per (experiment, system, condition), over the sources of
    ``queries`` filtered as the distributions run filters them (``{source: query or None}``)."""
    parts = []
    for source, query in queries.items():
        frame = attributes[attributes["source"] == source]
        if frame.empty:
            raise ValueError(f"no row of source {source!r} in the attribute table")
        parts.append(frame.query(query) if query else frame)
    rows = pd.concat(parts)
    if rows["condition"].isna().any():
        raise ValueError("rows without a condition among the counted sources")
    grouped = rows.groupby(["experiment", "system", "condition"])["failed"]
    return pd.DataFrame({"failed": grouped.sum().astype(int), "total": grouped.size()})


def failure_numbers(counts: pd.DataFrame,
                    exclude: list[tuple[str, str, str]]) -> dict[str, Decimal | int]:
    out: dict[str, Decimal | int] = {}
    rate = {}
    for (experiment, system, condition), row in counts.iterrows():
        stem = f"{EXPERIMENTS[experiment]}{SYSTEMS[system]}{macro_word(condition)}"
        out[f"failN{stem}"] = int(row["failed"])
        out[f"failTotal{stem}"] = int(row["total"])
        rate[experiment, system, condition] = round1(100 * row["failed"] / row["total"])
        out[f"failRate{stem}"] = rate[experiment, system, condition]
    excluded = {tuple(cell) for cell in exclude}
    out["failMaxRateOther"] = max(v for k, v in rate.items()
                                  if k[1] != NATURAL and k not in excluded)
    out["failMaxRateNatural"] = max(v for k, v in rate.items() if k[1] == NATURAL)
    return out
