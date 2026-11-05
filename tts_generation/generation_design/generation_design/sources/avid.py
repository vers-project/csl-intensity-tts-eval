"""AVID prompts and texts for the vocal-effort experiment.

Input is the AVID segment metadata with our hand annotation of the sessions merged in:
the AVID rows carry the segment columns (``signal_path``, ``segment_index``, ``start_s``,
``end_s``, ``text``, ``subject_id``, ``channel``) and, on the annotated speakers, the
annotation columns ``level``, ``excluded``, ``session_index``, ``sentence_id`` and
``repetition``.  The table may hold other corpora too.  Rows are restricted to one corpus and split (AVID test by default); VIC
rows carry a ``level`` of their own, so the corpus filter is not optional.

``sentence_id`` counts within a (speaker, session).  Takes of one sentence spread over
several sessions would be paired as if they were one reading, so that case is refused
and ``session_index`` must then be fixed in the config.

A *prompt set* is the four takes (one per level) of one sentence by one speaker.  Only
complete sets are used; an incomplete or ambiguous set is skipped and reported, never
completed with another take.

Target texts are the transcripts (what was said), so the text of a ``sentence_id`` is its
most frequent transcript; ``n_variants`` > 1 flags misreads or a ``sentence_id`` that does
not name the same sentence for every speaker.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ANNOTATION_COLUMNS = ("signal_path", "segment_index", "start_s", "end_s", "text",
                      "level", "sentence_id", "repetition", "excluded", "session_index")


def speaker_id(speaker) -> str:
    """``avid_10`` for a subject_id read back from CSV as ``10.0``."""
    return f"avid_{int(float(speaker))}"


def _as_bool(column: pd.Series) -> pd.Series:
    """CSV booleans: NA is False, and the strings "True"/"False" are not both truthy."""
    if column.dtype == bool:
        return column
    text = column.astype(object).where(column.notna(), "false").astype(str).str.strip().str.lower()
    unknown = sorted(set(text) - {"true", "false", "1", "0", "1.0", "0.0"})
    if unknown:
        raise ValueError(f"Unreadable boolean value(s) {unknown} in column {column.name!r}.")
    return text.isin(["true", "1", "1.0"])


def load_avid_annotated(metadata_csv: str | Path, *, repetition: int = 1,
                        speaker_column: str = "subject_id", corpus: str | None = "AVID",
                        split: str | None = "test",
                        session_index: int | None = None) -> pd.DataFrame:
    """Annotated speech rows of one corpus, split and repetition, with ``speaker_id`` added.

    ``corpus`` / ``split`` / ``session_index`` set to None keep every value.
    """
    table = pd.read_csv(metadata_csv, low_memory=False)
    missing = [c for c in (*ANNOTATION_COLUMNS, speaker_column) if c not in table.columns]
    if missing:
        raise ValueError(f"{metadata_csv} lacks column(s) {missing}.")
    for column, value in (("corpus", corpus), ("split", split)):
        if value is not None:
            if column not in table.columns:
                raise ValueError(f"{metadata_csv} has no {column!r} column to select "
                                 f"{value!r} on.")
            table = table[table[column] == value]
    excluded = _as_bool(table["excluded"])
    keep = (~excluded & table["level"].notna() & (table["level"] != "")
            & table["sentence_id"].notna() & (table["repetition"] == repetition))
    if session_index is not None:
        keep &= table["session_index"] == session_index
    rows = table[keep].copy()
    if rows.empty:
        raise ValueError(f"No annotated row in {metadata_csv} for corpus={corpus!r}, "
                         f"split={split!r}, repetition={repetition}.")
    rows["sentence_id"] = rows["sentence_id"].astype(int)
    rows["speaker_id"] = rows[speaker_column].map(speaker_id)

    sessions = rows.groupby(["speaker_id", "sentence_id"])["session_index"].nunique()
    spread = sessions[sessions > 1]
    if len(spread):
        raise ValueError(
            f"{len(spread)} (speaker, sentence_id) reading(s) span several sessions, e.g. "
            f"{spread.index[0]}; sentence_id counts within a session, so set "
            "avid.session_index in the config.")
    return rows


def avid_prompt_sets(
    annotated: pd.DataFrame,
    *,
    prompt_sentence_ids: list[int],
    levels: list[str],
    anchor: str,
    speakers: str | list = "all",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Complete prompt sets, and the (speaker, sentence) sets skipped with the reason.

    ``sets`` has one row per (speaker_id, prompt_sentence_id, condition) with the segment
    columns; ``skipped`` has speaker_id, prompt_sentence_id, reason.
    """
    if anchor not in levels:
        raise ValueError(f"anchor {anchor!r} is not one of the levels {levels}.")
    if speakers != "all":
        annotated = annotated[annotated["speaker_id"].isin([speaker_id(s) for s in speakers])]

    kept, skipped = [], []
    for spk in sorted(annotated["speaker_id"].unique()):
        for sid in prompt_sentence_ids:
            takes = annotated[(annotated["speaker_id"] == spk)
                              & (annotated["sentence_id"] == sid)]
            counts = takes["level"].value_counts()
            absent = [lvl for lvl in levels if counts.get(lvl, 0) == 0]
            doubled = [lvl for lvl in levels if counts.get(lvl, 0) > 1]
            if absent or doubled:
                reason = "; ".join(filter(None, [
                    f"missing level(s) {absent}" if absent else "",
                    f"several takes for level(s) {doubled}" if doubled else "",
                ]))
                skipped.append({"speaker_id": spk, "prompt_sentence_id": sid, "reason": reason})
                continue
            for lvl in levels:
                take = takes[takes["level"] == lvl].iloc[0]
                kept.append({**take.to_dict(), "prompt_sentence_id": sid, "condition": lvl})
    sets = pd.DataFrame(kept)
    return sets, pd.DataFrame(skipped, columns=["speaker_id", "prompt_sentence_id", "reason"])


def avid_target_texts(annotated: pd.DataFrame, target_sentence_ids: list[int]) -> pd.DataFrame:
    """sentence_id, text (most frequent transcript), n_takes, n_variants."""
    rows = []
    for sid in target_sentence_ids:
        transcripts = annotated.loc[annotated["sentence_id"] == sid, "text"].dropna().str.strip()
        transcripts = transcripts[transcripts != ""]
        if transcripts.empty:
            raise ValueError(f"No transcribed take for AVID sentence_id {sid}.")
        counts = transcripts.value_counts()
        rows.append({"sentence_id": sid, "text": counts.index[0],
                     "n_takes": int(counts.sum()), "n_variants": int(len(counts))})
    return pd.DataFrame(rows)


def cut_prompts(sets: pd.DataFrame, *, root: str | Path, output_dir: str | Path,
                channel: int = 0) -> pd.DataFrame:
    """Cut every prompt segment out of its session recording and write it as float32 WAV.

    Adds ``clip`` = path relative to ``output_dir``
    (``prompts/{speaker_id}/{prompt_sentence_id}/{condition}.wav``).  The row's own
    ``channel`` column wins over ``channel`` when present (0 = speech mic, 1 = EGG).
    """
    root, output_dir = Path(root), Path(output_dir)
    clips = []
    for row in sets.itertuples(index=False):
        source = root / row.signal_path
        sample_rate = sf.info(str(source)).samplerate
        start, stop = round(row.start_s * sample_rate), round(row.end_s * sample_rate)
        audio, _ = sf.read(str(source), start=start, stop=stop, dtype="float32", always_2d=True)
        row_channel = getattr(row, "channel", None)
        ch = channel if row_channel is None or pd.isna(row_channel) else int(row_channel)
        clip = Path("prompts") / row.speaker_id / str(row.prompt_sentence_id) / f"{row.condition}.wav"
        (output_dir / clip).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_dir / clip), np.ascontiguousarray(audio[:, ch]), sample_rate,
                 subtype="FLOAT")
        clips.append(str(clip))
    return sets.assign(clip=clips)
