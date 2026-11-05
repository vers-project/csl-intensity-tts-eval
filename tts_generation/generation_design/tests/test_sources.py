import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from conftest import DOGS, EMOTIONS, KIDS, LEVELS
from generation_design.sources import avid, ravdess

SENTENCES = {"kids": KIDS, "dogs": DOGS}


def test_ravdess_prompts_are_crossed(ravdess_csv):
    meta = ravdess.load_ravdess(ravdess_csv)
    clips = ravdess.ravdess_clips(meta, actors="all", sentences=SENTENCES,
                                  conditions=EMOTIONS, anchor="neutral")
    assert len(clips) == 2 * 2 * len(EMOTIONS)
    assert (clips["condition_clip_text"] != clips["text"]).all()
    kids = clips[clips["sentence_id"] == "kids"]
    assert (kids["condition_clip_text"] == DOGS).all()
    assert kids["condition_clip"].str.contains("-Dogs-1").all()


def test_ravdess_takes_strong_and_falls_back_to_normal_for_neutral(ravdess_csv):
    meta = ravdess.load_ravdess(ravdess_csv)
    clips = ravdess.ravdess_clips(meta, actors=[1], sentences=SENTENCES,
                                  conditions=EMOTIONS, anchor="neutral")
    angry = clips[clips["condition"] == "angry"]
    assert angry["condition_clip"].str.contains("-strong-").all()
    neutral = clips[clips["condition"] == "neutral"]
    assert neutral["condition_clip"].str.contains("neutral-normal").all()
    assert (clips["anchor_clip"].str.contains("neutral-normal")).all()


def test_ravdess_missing_take_raises(ravdess_csv):
    meta = ravdess.load_ravdess(ravdess_csv)
    meta = meta[~((meta["actor"] == 2) & (meta["emotion"] == "sad"))]
    with pytest.raises(ValueError, match="actor 2, emotion 'sad'"):
        ravdess.ravdess_clips(meta, actors="all", sentences=SENTENCES,
                              conditions=EMOTIONS, anchor="neutral")


def test_ravdess_rejects_non_statement_text(ravdess_csv):
    meta = ravdess.load_ravdess(ravdess_csv)
    with pytest.raises(ValueError, match="not RAVDESS statements"):
        ravdess.ravdess_clips(meta, actors="all", sentences={"a": KIDS, "b": "Hello."},
                              conditions=EMOTIONS, anchor="neutral")


def test_avid_keeps_only_complete_unambiguous_sets(avid_root):
    _, annotation = avid_root
    annotated = avid.load_avid_annotated(annotation, repetition=1)
    assert (annotated["repetition"] == 1).all()
    sets, skipped = avid.avid_prompt_sets(annotated, prompt_sentence_ids=[1, 2],
                                          levels=LEVELS, anchor="normal")
    kept = set(zip(sets["speaker_id"], sets["prompt_sentence_id"]))
    # Speaker 7 sentence 1 is complete once the excluded take is dropped; sentence 2 is not.
    assert kept == {("avid_3", 1), ("avid_3", 2), ("avid_7", 1)}
    assert list(skipped["speaker_id"]) == ["avid_7"]
    assert "missing level(s) ['loud']" in skipped.iloc[0]["reason"]
    assert len(sets) == 3 * len(LEVELS)


def test_avid_doubled_level_is_skipped(avid_root):
    _, annotation = avid_root
    table = pd.read_csv(annotation)
    table["excluded"] = False                     # the excluded soft take now doubles soft
    table.to_csv(annotation, index=False)
    sets, skipped = avid.avid_prompt_sets(avid.load_avid_annotated(annotation),
                                          prompt_sentence_ids=[1], levels=LEVELS,
                                          anchor="normal")
    assert set(sets["speaker_id"]) == {"avid_3"}
    assert "several takes for level(s) ['soft']" in skipped.iloc[0]["reason"]


def test_avid_excluded_read_as_text_is_not_truthy(avid_root):
    _, annotation = avid_root
    table = pd.read_csv(annotation)
    table["excluded"] = table["excluded"].map({True: "True", False: "False"})
    table.to_csv(annotation, index=False)
    annotated = avid.load_avid_annotated(annotation)
    assert len(annotated) == len(table[(table["excluded"] == "False")
                                       & (table["repetition"] == 1)
                                       & (table["corpus"] == "AVID")])


def test_avid_reads_only_annotated_avid_test_rows(avid_root):
    _, annotation = avid_root
    annotated = avid.load_avid_annotated(annotation)
    assert set(annotated["corpus"]) == {"AVID"} and set(annotated["split"]) == {"test"}
    # subject_id is read back as a float; the speaker id must not say "3.0".
    assert set(annotated["speaker_id"]) == {"avid_3", "avid_7"}


def test_avid_reading_spread_over_sessions_is_refused(avid_root):
    _, annotation = avid_root
    table = pd.read_csv(annotation)
    second = (table["subject_id"] == 3) & (table["sentence_id"] == 2) & (table["level"] == "loud")
    table.loc[second, "session_index"] = 2
    table.to_csv(annotation, index=False)
    with pytest.raises(ValueError, match="span several sessions"):
        avid.load_avid_annotated(annotation)
    annotated = avid.load_avid_annotated(annotation, session_index=1)
    assert (annotated["session_index"] == 1).all()


def test_avid_target_text_is_most_frequent_transcript(avid_root):
    _, annotation = avid_root
    annotated = avid.load_avid_annotated(annotation)
    targets = avid.avid_target_texts(annotated, [5, 6]).set_index("sentence_id")
    assert targets.at[5, "text"] == "Sentence number 5."
    assert targets.at[5, "n_variants"] == 2
    assert targets.at[6, "n_variants"] == 1
    with pytest.raises(ValueError, match="sentence_id 9"):
        avid.avid_target_texts(annotated, [9])


def test_cut_prompts_reads_speech_channel_as_float(avid_root, tmp_path):
    root, annotation = avid_root
    annotated = avid.load_avid_annotated(annotation)
    sets, _ = avid.avid_prompt_sets(annotated, prompt_sentence_ids=[1], levels=LEVELS,
                                    anchor="normal", speakers=[3])
    out = tmp_path / "job"
    cut = avid.cut_prompts(sets, root=root, output_dir=out)
    audio, sr = sf.read(str(out / cut.iloc[0]["clip"]), dtype="float32")
    assert sf.info(str(out / cut.iloc[0]["clip"])).subtype == "FLOAT"
    assert audio.ndim == 1 and len(audio) == int(0.5 * sr)
    np.testing.assert_allclose(audio, 0.5)
