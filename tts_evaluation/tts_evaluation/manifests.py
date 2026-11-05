"""Evaluation manifests: one table per source, read unchanged by the predictor and by speech-eval.

A *source* is one generation run (system × experiment: ``cosyvoice3_emotion`` …) or one
natural reference corpus (``ravdess``, ``avid``).  Each row is one audio item, with:

    utt_id, signal_path, group_id, condition   speech-eval's required columns
    corpus, text, start_s, end_s, channel      read by both consumers; ``corpus`` is the
                                               key into the configs' ``dataset_roots``
    sex                                        ``F``/``M`` (see ``speaker_sex``)
    source, system, experiment, seed, speaker_id, sentence_id, prompt_sentence_id,
    pair_id, is_anchor                         what the analysis joins and groups on
    untrimmed_start_s … vad_status             the outer-silence trim (TRIM_COLUMNS)

Paths stay relative to a dataset root.  A TTS row's ``corpus`` is its system and its
``signal_path`` starts at the experiment folder (``emotion/<date>/<time>/<job>/audio/…``),
so the run folders are named once, in the manifest config, and the consumers' configs need
one root per system.

Every row is trimmed of its leading and trailing silence (``audio_utils.data.vad``), so the
systems' different silence padding does not move level-based measures.  A row where the VAD
finds no speech is kept and flagged (``vad_status == "no_speech"``), never dropped here.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from audio_utils.data.metadata import resolve_paths
from audio_utils.data.vad import trim_metadata_with_vad
from generation_design.sources import avid, ravdess

ID_COLUMNS = [
    "utt_id", "signal_path", "group_id", "condition",
    "corpus", "text", "start_s", "end_s", "channel", "sex",
    "source", "system", "experiment", "seed", "speaker_id", "sentence_id",
    "prompt_sentence_id", "pair_id", "is_anchor",
]


def _ordered(rows: pd.DataFrame) -> pd.DataFrame:
    """The identity columns first, every other column after them in its own order."""
    for column in ID_COLUMNS:
        if column not in rows.columns:
            rows[column] = np.nan
    rest = [c for c in rows.columns if c not in ID_COLUMNS]
    return rows[ID_COLUMNS + rest].reset_index(drop=True)


def normalise_sex(value) -> str | float:
    """``F``/``M`` from any spelling (``female``, ``M``, ``Male``); NaN if unknown."""
    key = str(value).strip().upper()[:1] if pd.notna(value) else ""
    return key if key in ("F", "M") else np.nan


# Qwen3-TTS's designed voices are named by sex and age in its generation configs (F20, M30).
VOICE_ID = re.compile(r"^([FM])\d+$")


def speaker_sex(*natural: pd.DataFrame) -> dict[str, str]:
    """``{speaker_id: sex}`` from the natural sources' rows; one sex per speaker or refused."""
    pairs = pd.concat([rows[["speaker_id", "sex"]] for rows in natural]).dropna()
    per_speaker = pairs.groupby("speaker_id")["sex"].unique()
    conflicting = per_speaker[per_speaker.map(len) > 1]
    if len(conflicting):
        raise ValueError(f"Speaker(s) with more than one sex: {dict(conflicting)}.")
    return per_speaker.map(lambda values: values[0]).to_dict()


def tts_sex(speaker_ids: pd.Series, sex_of: dict[str, str]) -> pd.Series:
    """The sex of the speaker each sample clones (natural rows), else its voice id's."""
    def one(speaker):
        if speaker in sex_of:
            return sex_of[speaker]
        match = VOICE_ID.match(str(speaker))
        if match:
            return match.group(1)
        raise ValueError(f"No sex for TTS speaker {speaker!r}: not a natural speaker and "
                         f"not a voice id like F20.")
    return speaker_ids.map(one)


def latest_status(log: pd.DataFrame) -> pd.Series:
    """Status of each ``output_path``'s last attempt (the log appends one line per attempt)."""
    return log.groupby("output_path", sort=False)["status"].last()


def tts_rows(system: str, experiment: str, run_folder: str | Path,
             system_root: str | Path, sex_of: dict[str, str] | None = None) -> pd.DataFrame:
    """The generated samples of one run whose last generation attempt is ``ok``.

    ``run_folder`` is relative to ``<system_root>/<experiment>/`` (``10-15-26/17-26-43``)
    and holds one sub-folder per array job, each with ``manifest.csv`` and
    ``generation_log.csv``.  A manifest row with no log line means the job never reached
    it, which is refused rather than silently dropped.  ``sex_of`` is ``speaker_sex`` of
    the natural sources (see ``tts_sex``).
    """
    run_dir = Path(system_root) / experiment / run_folder
    jobs = sorted((p for p in run_dir.iterdir() if (p / "manifest.csv").is_file()),
                  key=lambda p: p.name)
    if not jobs:
        raise FileNotFoundError(f"No job folder with a manifest.csv under {run_dir}.")

    tables = []
    for job in jobs:
        manifest = pd.read_csv(job / "manifest.csv")
        status = latest_status(pd.read_csv(job / "generation_log.csv"))
        unlogged = sorted(set(manifest["output_path"]) - set(status.index))
        if unlogged:
            raise ValueError(f"{len(unlogged)} manifest row(s) of {job} have no line in "
                             f"generation_log.csv, e.g. {unlogged[0]}.")
        manifest["generation_status"] = manifest["output_path"].map(status)
        manifest["job"] = job.name
        manifest["signal_path"] = [
            str(Path(experiment) / run_folder / job.name / path)
            for path in manifest["output_path"]]
        tables.append(manifest)
    rows = pd.concat(tables, ignore_index=True)

    failed = rows["generation_status"] != "ok"
    if failed.any():
        print(f"[{system}_{experiment}] {int(failed.sum())}/{len(rows)} rows not ok, dropped.")
    rows = rows[~failed].copy()

    rows["source"] = f"{system}_{experiment}"
    rows["corpus"] = system
    rows["utt_id"] = [f"{system}|{experiment}|{pair}|{condition}"
                      for pair, condition in zip(rows["pair_id"], rows["condition"])]
    rows["group_id"] = rows["pair_id"]
    rows["sex"] = tts_sex(rows["speaker_id"], sex_of or {})
    return _ordered(rows)


def ravdess_rows(metadata_csv: str | Path, sentences: dict[str, str]) -> pd.DataFrame:
    """Every RAVDESS clip, with ``sentence_id`` named as in the emotion generation configs.

    ``group_id`` is the parallel unit (actor × statement × repetition): its rows differ only
    in emotion and intensity.
    """
    rows = ravdess.load_ravdess(metadata_csv).copy()
    sentence_of = {text: sentence_id for sentence_id, text in sentences.items()}
    unknown = sorted(set(rows["statement"]) - set(sentence_of))
    if unknown:
        raise ValueError(f"RAVDESS statement(s) {unknown} have no id in `sentences`.")

    rows["source"] = "ravdess"
    rows["corpus"] = "RAVDESS"
    rows["system"] = "natural"
    rows["experiment"] = "emotion"
    rows["utt_id"] = rows["signal_path"]
    rows["speaker_id"] = rows["actor"].map(ravdess.speaker_id)
    rows["sex"] = rows["gender"].map(normalise_sex)
    rows["sentence_id"] = rows["statement"].map(sentence_of)
    rows["text"] = rows["statement"]
    rows["condition"] = rows["emotion"]
    rows["is_anchor"] = rows["emotion"] == "neutral"
    rows["group_id"] = [f"{speaker}|{sentence}|{int(repetition)}" for speaker, sentence,
                        repetition in zip(rows["speaker_id"], rows["sentence_id"],
                                          rows["repetition"])]
    rows["pair_id"] = rows["group_id"]
    return _ordered(rows)


def avid_rows(metadata_csv: str | Path, *, corpus: str = "AVID", split: str = "test",
              anchor: str = "normal", channel: int = 0) -> pd.DataFrame:
    """Every row of one AVID split, every metadata column kept (calibration included).

    The split holds both the effort prompts and the natural targets.  ``group_id`` = speaker × session × repetition ×
    sentence, the unit whose takes differ only in effort level; it is empty on rows
    without the hand annotation.  ``excluded`` takes are kept and left to the analysis.
    """
    table = pd.read_csv(metadata_csv, low_memory=False)
    rows = table[(table["corpus"] == corpus) & (table["split"] == split)].copy()
    if rows.empty:
        raise ValueError(f"No row of {metadata_csv} has corpus={corpus!r}, split={split!r}.")

    rows["source"] = "avid"
    rows["system"] = "natural"
    rows["experiment"] = "effort"
    rows["utt_id"] = [f"{path}|{int(index)}"
                      for path, index in zip(rows["signal_path"], rows["segment_index"])]
    rows["speaker_id"] = rows["subject_id"].map(avid.speaker_id)
    rows["sex"] = rows["sex"].map(normalise_sex)
    rows["condition"] = rows["level"]
    rows["is_anchor"] = rows["level"] == anchor
    rows["channel"] = rows["channel"].fillna(channel) if "channel" in rows else channel

    annotated = rows["level"].notna() & rows["sentence_id"].notna()
    rows["sentence_id"] = rows["sentence_id"].astype("Int64")
    rows["group_id"] = pd.Series(
        [f"{speaker}|{int(session)}|{int(repetition)}|{int(sentence)}" if keep else np.nan
         for keep, speaker, session, repetition, sentence in zip(
             annotated, rows["speaker_id"], rows["session_index"].fillna(0),
             rows["repetition"].fillna(0), rows["sentence_id"].fillna(0))],
        index=rows.index, dtype=object)
    rows["pair_id"] = rows["group_id"]
    return _ordered(rows)


def check_unique_utt_ids(rows: pd.DataFrame, source: str) -> None:
    duplicated = rows["utt_id"][rows["utt_id"].duplicated()].unique()
    if len(duplicated):
        raise ValueError(f"[{source}] {len(duplicated)} duplicated utt_id, e.g. "
                         f"{duplicated[0]!r}.")


def trim(rows: pd.DataFrame, dataset_roots: dict[str, str | Path],
         vad_params: dict | None = None, progress: bool = True) -> pd.DataFrame:
    """``rows`` with ``start_s``/``end_s`` moved onto the speech; paths stay relative."""
    resolved = resolve_paths(rows, dataset_roots)
    trimmed = trim_metadata_with_vad(resolved, vad_params, progress=progress)
    trimmed["signal_path"] = rows["signal_path"].to_numpy()
    return trimmed


def summarise(rows: pd.DataFrame) -> dict:
    """One line of ``summary.csv``: size and what the trim did."""
    ok = rows["vad_status"] == "ok"
    return {
        "source": rows["source"].iloc[0],
        "rows": len(rows),
        "no_speech": int((~ok).sum()),
        "median_head_s": float(rows.loc[ok, "trimmed_head_s"].median()),
        "median_tail_s": float(rows.loc[ok, "trimmed_tail_s"].median()),
        "median_kept_s": float((rows.loc[ok, "end_s"] - rows.loc[ok, "start_s"]).median()),
    }
