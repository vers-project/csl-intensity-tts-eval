"""Config -> manifest: the parallel design of one generation job (one system, one seed).

The manifest has one row per WAV file to generate.  Rows sharing a ``pair_id`` (speaker,
prompt sentence, target sentence, seed) differ only in ``condition``; any join uses
``pair_id``, never row position.

Three ways a config names its speakers:

    ravdess:  block   RAVDESS actors, condition + anchor clips  (emotion; CosyVoice 3, IndexTTS2)
    avid:     block   AVID speakers with prompt_sentence_ids   (effort;  CosyVoice 3, IndexTTS2)
    speakers: mapping {voice_id: voice description}, no clip  (Qwen3-TTS, both experiments)
    speakers: list    named voices, no clip

Each system reads the two clip columns its own way (CosyVoice 3: prompt = condition_clip;
IndexTTS2: speaker prompt = anchor_clip, emotion prompt = condition_clip in effort).

``control`` holds the value of a ``conditions`` mapping (IndexTTS2 slot, Qwen3-TTS phrase),
formatted through ``template`` when the config has one (``{name}`` = speaker id, ``{voice}``
= its description, or the id for a list, ``{phrase}`` = the condition value), so a dry run
shows every description in full.

Clip paths are relative to ``clip_root``: the corpus root for RAVDESS, ``.`` (the job's
output directory) for the AVID prompts this module cuts.  ``output_path`` is relative to the
output directory.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from generation_design.sources import avid, ravdess

MANIFEST_COLUMNS = [
    "pair_id", "system", "experiment", "seed",
    "speaker_id", "prompt_sentence_id", "sentence_id", "text",
    "condition", "is_anchor", "control",
    "clip_root", "condition_clip", "condition_clip_text", "anchor_clip", "anchor_clip_text",
    "output_path",
]


def build_manifest(config: dict, *, system: str, output_dir: str | Path) -> pd.DataFrame:
    """Expand ``config`` into the manifest, check it, write ``manifest.csv`` and return it.

    Effort configs also write ``target_texts.csv`` and, when AVID prompts are used,
    the cut prompts under ``prompts/``, ``skipped_prompt_sets.csv``, and
    ``prompt_sets.csv``: the prompts' own metadata rows (``clip`` = the cut WAV), so the
    prompts can be measured with the same pipeline as the outputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = list(config["conditions"])
    anchor = config["anchor"]
    if anchor not in conditions:
        raise ValueError(f"anchor {anchor!r} is not one of the conditions {conditions}.")

    if "ravdess" in config:
        rows = _ravdess_rows(config, conditions, anchor)
    elif "avid" in config and "prompt_sentence_ids" in config["avid"]:
        rows = _avid_prompt_rows(config, conditions, anchor, output_dir)
    elif "speakers" in config:
        rows = _named_voice_rows(config, conditions, output_dir)
    else:
        raise ValueError("The config names no speakers: expected a 'ravdess' block, an "
                         "'avid' block with prompt_sentence_ids, or a 'speakers' list.")

    manifest = _complete(rows, config, system=system, anchor=anchor)
    check_manifest(manifest, conditions)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    _print_summary(manifest)
    return manifest


def check_manifest(manifest: pd.DataFrame, conditions: list[str]) -> None:
    """Raise if any pair is broken.  Every rule here is part of the protocol."""
    problems = []
    by_pair = manifest.groupby("pair_id")["condition"].agg(lambda c: sorted(c))
    broken = by_pair[by_pair.map(lambda c: c != sorted(conditions))]
    if len(broken):
        problems.append(f"{len(broken)} pair(s) without exactly one sample per condition, "
                        f"e.g. {broken.index[0]!r}: {broken.iloc[0]}")
    if manifest["output_path"].duplicated().any():
        problems.append("duplicated output_path")
    with_clip = manifest[manifest["clip_root"].notna()]
    if with_clip[["condition_clip", "anchor_clip"]].isna().any().any():
        problems.append("rows with a clip_root but no condition or anchor clip")
    copied = with_clip[(with_clip["condition_clip_text"] == with_clip["text"])
                       | (with_clip["anchor_clip_text"] == with_clip["text"])]
    if len(copied):
        problems.append(f"{len(copied)} row(s) whose prompt says the target sentence, e.g. "
                        f"{copied.iloc[0]['pair_id']!r}")
    same_sentence = manifest["prompt_sentence_id"].notna() & (
        manifest["prompt_sentence_id"].astype(str) == manifest["sentence_id"].astype(str))
    if same_sentence.any():
        problems.append("target sentence equal to the prompt sentence")
    if problems:
        raise ValueError("Broken manifest:\n  " + "\n  ".join(problems))


def _ravdess_rows(config: dict, conditions: list[str], anchor: str) -> pd.DataFrame:
    block = config["ravdess"]
    meta = ravdess.load_ravdess(block["metadata_csv"])
    rows = ravdess.ravdess_clips(
        meta,
        actors=block.get("actors", "all"),
        sentences=config["sentences"],
        conditions=conditions,
        anchor=anchor,
        repetition=block.get("repetition", 1),
        intensity=block.get("intensity", "strong"),
    )
    return rows.assign(clip_root=str(block["root"]), prompt_sentence_id=pd.NA)


def _avid_annotated(block: dict) -> pd.DataFrame:
    return avid.load_avid_annotated(block["metadata_csv"],
                                    repetition=block.get("repetition", 1),
                                    speaker_column=block.get("speaker_column", "subject_id"),
                                    corpus=block.get("corpus", "AVID"),
                                    split=block.get("split", "test"),
                                    session_index=block.get("session_index"))


def _avid_targets(block: dict, annotated: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    targets = avid.avid_target_texts(annotated, block["target_sentence_ids"])
    targets.to_csv(output_dir / "target_texts.csv", index=False)
    return targets


def _avid_prompt_rows(config: dict, conditions: list[str], anchor: str,
                      output_dir: Path) -> pd.DataFrame:
    block = config["avid"]
    overlap = sorted(set(block["prompt_sentence_ids"]) & set(block["target_sentence_ids"]))
    if overlap:
        raise ValueError(f"sentence_id(s) {overlap} are both prompt and target sentences.")
    annotated = _avid_annotated(block)
    sets, skipped = avid.avid_prompt_sets(
        annotated, prompt_sentence_ids=block["prompt_sentence_ids"], levels=conditions,
        anchor=anchor, speakers=block.get("speakers", "all"))
    skipped.to_csv(output_dir / "skipped_prompt_sets.csv", index=False)
    if sets.empty:
        raise ValueError("No complete AVID prompt set; see skipped_prompt_sets.csv.")
    sets = avid.cut_prompts(sets, root=block["root"], output_dir=output_dir,
                            channel=block.get("channel", 0))
    sets.to_csv(output_dir / "prompt_sets.csv", index=False)
    targets = _avid_targets(block, annotated, output_dir)

    rows = []
    for (spk, prompt_id), takes in sets.groupby(["speaker_id", "prompt_sentence_id"]):
        by_level = takes.set_index("condition")
        for target in targets.itertuples(index=False):
            for condition in conditions:
                rows.append({
                    "speaker_id": spk,
                    "prompt_sentence_id": prompt_id,
                    "sentence_id": target.sentence_id,
                    "text": target.text,
                    "condition": condition,
                    "condition_clip": by_level.at[condition, "clip"],
                    "condition_clip_text": by_level.at[condition, "text"],
                    "anchor_clip": by_level.at[anchor, "clip"],
                    "anchor_clip_text": by_level.at[anchor, "text"],
                    "clip_root": ".",
                })
    return pd.DataFrame(rows)


def _named_voice_rows(config: dict, conditions: list[str], output_dir: Path) -> pd.DataFrame:
    if config["experiment"] == "effort":
        block = config["avid"]
        targets = _avid_targets(block, _avid_annotated(block), output_dir)
        sentences = dict(zip(targets["sentence_id"], targets["text"]))
    else:
        sentences = config["sentences"]
    rows = [{"speaker_id": name, "sentence_id": sentence_id, "text": text, "condition": c}
            for name in config["speakers"]
            for sentence_id, text in sentences.items()
            for c in conditions]
    return pd.DataFrame(rows).assign(prompt_sentence_id=pd.NA, clip_root=pd.NA,
                                     condition_clip=pd.NA, condition_clip_text=pd.NA,
                                     anchor_clip=pd.NA, anchor_clip_text=pd.NA)


def _complete(rows: pd.DataFrame, config: dict, *, system: str, anchor: str) -> pd.DataFrame:
    seed = int(config["seed"])
    rows = rows.assign(system=system, experiment=config["experiment"], seed=seed,
                       is_anchor=rows["condition"] == anchor)
    rows["control"] = [_control(config, r.condition, r.speaker_id) for r in rows.itertuples()]
    prompt = rows["prompt_sentence_id"].map(lambda p: "-" if pd.isna(p) else str(p))
    rows["pair_id"] = (rows["speaker_id"].astype(str) + "|" + prompt + "|"
                       + rows["sentence_id"].astype(str) + "|" + str(seed))
    folders = rows["speaker_id"].astype(str) + prompt.map(lambda p: "" if p == "-" else f"/{p}")
    rows["output_path"] = ("audio/" + folders + "/" + rows["sentence_id"].astype(str) + "/"
                           + rows["condition"] + ".wav")
    return rows[MANIFEST_COLUMNS].reset_index(drop=True)


def _control(config: dict, condition: str, speaker: str):
    conditions = config["conditions"]
    if not isinstance(conditions, dict) or conditions[condition] is None:
        return pd.NA
    value = conditions[condition]
    if "template" in config:
        voices = config.get("speakers")
        voice = voices[speaker] if isinstance(voices, dict) else speaker
        return config["template"].format(name=speaker, voice=voice, phrase=value)
    return value


def _print_summary(manifest: pd.DataFrame) -> None:
    print(f"Manifest: {len(manifest)} samples, {manifest['pair_id'].nunique()} pairs, "
          f"{manifest['speaker_id'].nunique()} speakers, "
          f"{manifest['sentence_id'].nunique()} target sentences, "
          f"{manifest['condition'].nunique()} conditions.")
