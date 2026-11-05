"""Attribute table: one row per utterance, every attribute, and the failure screen.

Per source, the evaluation manifest is joined with the predictor's output
(``predictions/utterances.csv``) and the speech-eval metrics (``metrics.csv``) on
``utt_id``. The predictor's columns are renamed ``pphi.<key>``, after speech-eval's
``<metric>.<key>`` naming.

A sample is a failed generation when the predictor finds no speech in it, on TTS and
natural rows alike:

    fail.no_speech   no speech frame (``pphi.n_frames == 0``)
    failed           any ``fail.*`` column
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PPHI_COLUMNS = {"n_frames": "pphi.n_frames", "leq_pred": "pphi.leq_db",
                "leq_label": "pphi.leq_label_db"}
FAIL_COLUMNS = ["fail.no_speech"]


def run_jobs(run_dir: str | Path) -> dict[str, Path]:
    """``{source: job folder}`` of a launcher sweep, from its ``sweep_lookup.csv``.

    The source is the stem of the manifest each job read, so the mapping never relies on
    the sweep's order.
    """
    run_dir = Path(run_dir)
    lookup = pd.read_csv(run_dir / "sweep_lookup.csv")
    columns = [c for c in lookup.columns if c in ("data.manifest", "data.metadata_csv")]
    if len(columns) != 1:
        raise KeyError(f"{run_dir}/sweep_lookup.csv names no single manifest column: "
                       f"{list(lookup.columns)}")
    return {Path(manifest).stem: run_dir / str(job)
            for job, manifest in zip(lookup["job_index"], lookup[columns[0]])}


def _check_same_ids(manifest: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    if other["utt_id"].duplicated().any():
        raise ValueError(f"{name}: duplicated utt_id.")
    missing = set(manifest["utt_id"]) - set(other["utt_id"])
    extra = set(other["utt_id"]) - set(manifest["utt_id"])
    if missing or extra:
        raise ValueError(f"{name}: {len(missing)} manifest utt_id missing, {len(extra)} "
                         f"not in the manifest.")


def load_source(manifest_csv: str | Path, pphi_dir: str | Path, metrics_dir: str | Path
                ) -> pd.DataFrame:
    """The joined table of one source."""
    manifest = pd.read_csv(manifest_csv, low_memory=False)
    if manifest["utt_id"].duplicated().any():
        raise ValueError(f"{manifest_csv}: duplicated utt_id.")

    pphi = pd.read_csv(Path(pphi_dir) / "predictions" / "utterances.csv", low_memory=False)
    _check_same_ids(manifest, pphi, f"{pphi_dir} (P_φ)")
    pphi = pphi[["utt_id", *[c for c in PPHI_COLUMNS if c in pphi.columns]]]
    pphi = pphi.rename(columns=PPHI_COLUMNS)

    metrics = pd.read_csv(Path(metrics_dir) / "metrics.csv", low_memory=False)
    _check_same_ids(manifest, metrics, f"{metrics_dir} (speech-eval)")
    metrics = metrics[["utt_id", *[c for c in metrics.columns if c not in manifest.columns]]]

    return (manifest.merge(pphi, on="utt_id", validate="one_to_one")
                    .merge(metrics, on="utt_id", validate="one_to_one"))


def drop_excluded(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Drop the rows a corpus marks ``excluded`` (AVID's unlevelled takes)."""
    if "excluded" in frame.columns:
        excluded = frame["excluded"].fillna(False).astype(bool)
        if excluded.any():
            print(f"[{source}] {int(excluded.sum())} excluded rows dropped.")
        frame = frame[~excluded]
    if frame["condition"].isna().any():
        raise ValueError(f"[{source}] {int(frame['condition'].isna().sum())} rows have no "
                         "condition.")
    return frame.reset_index(drop=True)


def flag_failures(frame: pd.DataFrame) -> pd.DataFrame:
    """The ``fail.*`` columns and ``failed``."""
    frame = frame.copy()
    frame["fail.no_speech"] = frame["pphi.n_frames"] == 0
    frame["failed"] = frame[FAIL_COLUMNS].any(axis=1)
    return frame


def failure_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Rows and failures per reason, per source × condition."""
    grouped = table.groupby(["source", "condition"], sort=False)
    summary = grouped[FAIL_COLUMNS + ["failed"]].sum().astype(int)
    summary.insert(0, "n", grouped.size())
    return summary.reset_index()


def coverage(table: pd.DataFrame, attributes: dict[str, str]) -> pd.DataFrame:
    """NaN count per source for each analysis attribute (``{name: column}``)."""
    missing = [c for c in attributes.values() if c not in table.columns]
    if missing:
        raise KeyError(f"Attribute columns not in the table: {missing}")
    nan = table[list(attributes.values())].isna()
    nan.columns = list(attributes)
    counts = nan.groupby(table["source"], sort=False).sum().astype(int)
    counts.insert(0, "n", table.groupby("source", sort=False).size())
    return counts.reset_index()
