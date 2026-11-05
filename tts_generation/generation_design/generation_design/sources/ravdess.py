"""RAVDESS clips for the emotion experiment.

Input is a RAVDESS metadata CSV (``signal_path``
relative to the RAVDESS root, ``actor``, ``emotion``, ``intensity``, ``statement`` as text,
``repetition``).

For every (actor, target sentence, condition) one *condition clip* is chosen — the actor's
take of that emotion — together with the actor's *anchor clip*.  Both clips are always
taken from the statement that is NOT being synthesised (crossed prompts), so a system never
hears its own target sentence in the prompt; this needs exactly the two RAVDESS statements.

A clip is never substituted: if the requested take does not exist, this raises.  The only
fallback is RAVDESS's own design: an emotion recorded at ``normal`` intensity only
(neutral) is taken at ``normal`` when ``strong`` is requested.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("signal_path", "actor", "emotion", "intensity", "statement", "repetition")


def load_ravdess(metadata_csv: str | Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in meta.columns]
    if missing:
        raise ValueError(f"{metadata_csv} lacks column(s) {missing}.")
    return meta


def speaker_id(actor: int) -> str:
    return f"actor_{int(actor):02d}"


def _pick(meta: pd.DataFrame, actor: int, emotion: str, statement: str,
          repetition: int, intensity: str) -> pd.Series:
    take = meta[(meta["actor"] == actor) & (meta["emotion"] == emotion)
                & (meta["statement"] == statement) & (meta["repetition"] == repetition)]
    if take.empty:
        raise ValueError(f"RAVDESS has no take for actor {actor}, emotion {emotion!r}, "
                         f"statement {statement!r}, repetition {repetition}.")
    at_intensity = take[take["intensity"] == intensity]
    if at_intensity.empty and set(take["intensity"]) == {"normal"}:
        at_intensity = take
    if len(at_intensity) != 1:
        raise ValueError(f"Expected one RAVDESS take for actor {actor}, emotion {emotion!r}, "
                         f"statement {statement!r}, repetition {repetition}, intensity "
                         f"{intensity!r}; found {len(at_intensity)}.")
    return at_intensity.iloc[0]


def ravdess_clips(
    meta: pd.DataFrame,
    *,
    actors: str | list[int],
    sentences: dict[str, str],
    conditions: list[str],
    anchor: str,
    repetition: int = 1,
    intensity: str = "strong",
) -> pd.DataFrame:
    """One row per (actor, sentence, condition) with its condition clip and anchor clip.

    Columns: speaker_id, sentence_id, text, condition, condition_clip, condition_clip_text,
    anchor_clip, anchor_clip_text.  Clip paths are ``signal_path`` (relative to the root).
    """
    if len(sentences) != 2:
        raise ValueError(f"Crossed prompts need exactly two sentences, got {len(sentences)}.")
    absent_texts = sorted(set(sentences.values()) - set(meta["statement"]))
    if absent_texts:
        raise ValueError(f"Sentence(s) {absent_texts} are not RAVDESS statements.")
    if anchor not in conditions:
        raise ValueError(f"anchor {anchor!r} is not one of the conditions {conditions}.")
    unknown = sorted(set(conditions) - set(meta["emotion"]))
    if unknown:
        raise ValueError(f"Condition(s) {unknown} are not RAVDESS emotions.")

    all_actors = sorted(meta["actor"].unique())
    if actors == "all":
        actors = all_actors
    absent = sorted(set(actors) - set(all_actors))
    if absent:
        raise ValueError(f"Actor(s) {absent} are not in the RAVDESS metadata.")

    texts = list(sentences.values())
    rows = []
    for actor in actors:
        for sentence_id, text in sentences.items():
            clip_text = texts[1] if text == texts[0] else texts[0]
            anchor_take = _pick(meta, actor, anchor, clip_text, repetition, intensity)
            for condition in conditions:
                take = _pick(meta, actor, condition, clip_text, repetition, intensity)
                rows.append({
                    "speaker_id": speaker_id(actor),
                    "sentence_id": sentence_id,
                    "text": text,
                    "condition": condition,
                    "condition_clip": take["signal_path"],
                    "condition_clip_text": clip_text,
                    "anchor_clip": anchor_take["signal_path"],
                    "anchor_clip_text": clip_text,
                })
    return pd.DataFrame(rows)
