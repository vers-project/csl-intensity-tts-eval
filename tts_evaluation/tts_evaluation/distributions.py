"""Absolute attribute distributions per condition: each system treated as a speaker.

No difference is taken: every usable utterance contributes its value, so systems and natural
speech are compared on the range they produce, whatever their input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def describe(frame: pd.DataFrame, attributes: dict[str, str], by: Sequence[str],
             quantiles: Sequence[float]) -> pd.DataFrame:
    """Per ``by`` × attribute: ``n, n_speakers, mean, sd`` and one ``q<percent>`` column per
    quantile. Failed rows and NaN values are left out."""
    missing = [c for c in attributes.values() if c not in frame.columns]
    if missing:
        raise KeyError(f"Attribute columns not in the table: {missing}")
    rows = frame[~frame["failed"].astype(bool)]
    long = pd.concat([rows[[*by, "speaker_id"]].assign(attribute=name,
                                                       value=rows[column].to_numpy(dtype=float))
                      for name, column in attributes.items()], ignore_index=True)
    grouped = long.dropna(subset=["value"]).groupby([*by, "attribute"], sort=False)
    out = grouped["value"].agg(n="size", mean="mean", sd="std")
    out.insert(1, "n_speakers", grouped["speaker_id"].nunique())
    for q in quantiles:
        out[f"q{round(100 * q):02d}"] = grouped["value"].quantile(q)
    return out.reset_index()


def join_utterances(frame: pd.DataFrame, utterances_csv: str | Path, name: str,
                    columns: dict[str, str]) -> pd.DataFrame:
    """``frame`` with ``columns`` ({column: key}) of another run's ``utterances.csv``, joined
    on ``utt_id`` and named ``<name>.<key>``. Every row of ``frame`` must be in the run."""
    other = pd.read_csv(utterances_csv, usecols=["utt_id", *columns], low_memory=False)
    if other["utt_id"].duplicated().any():
        raise ValueError(f"{utterances_csv}: duplicated utt_id.")
    missing = set(frame["utt_id"]) - set(other["utt_id"])
    if missing:
        raise ValueError(f"{utterances_csv}: {len(missing)} utt_id of the table missing.")
    other = other.rename(columns={c: f"{name}.{key}" for c, key in columns.items()})
    return frame.merge(other, on="utt_id", how="left", validate="one_to_one")
