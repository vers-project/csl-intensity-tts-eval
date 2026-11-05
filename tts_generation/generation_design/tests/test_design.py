import pandas as pd
import pytest

from conftest import DOGS, EMOTIONS, KIDS, LEVELS
from generation_design.design import build_manifest, check_manifest

SENTENCES = {"kids": KIDS, "dogs": DOGS}


def emotion_config(ravdess_csv, **overrides):
    config = {
        "experiment": "emotion", "seed": 2, "anchor": "neutral", "conditions": EMOTIONS,
        "sentences": SENTENCES,
        "ravdess": {"metadata_csv": str(ravdess_csv), "root": "/data/RAVDESS",
                    "actors": "all", "repetition": 1, "intensity": "strong"},
    }
    config.update(overrides)
    return config


def effort_config(root, annotation, **avid_overrides):
    block = {"root": str(root), "metadata_csv": str(annotation), "repetition": 1,
             "prompt_sentence_ids": [1, 2], "target_sentence_ids": [4, 5, 6]}
    block.update(avid_overrides)
    return {"experiment": "effort", "seed": 0, "anchor": "normal", "conditions": LEVELS,
            "avid": block}


def test_emotion_manifest_is_parallel(ravdess_csv, tmp_path):
    manifest = build_manifest(emotion_config(ravdess_csv), system="cosyvoice3",
                              output_dir=tmp_path)
    assert len(manifest) == 2 * 2 * len(EMOTIONS)
    assert manifest["pair_id"].nunique() == 4
    assert manifest.groupby("pair_id")["is_anchor"].sum().eq(1).all()
    assert (manifest["seed"] == 2).all()
    assert manifest.iloc[0]["pair_id"] == "actor_01|-|kids|2"
    assert manifest.iloc[0]["output_path"] == "audio/actor_01/kids/neutral.wav"
    assert manifest["control"].isna().all()
    written = pd.read_csv(tmp_path / "manifest.csv")
    assert list(written.columns) == list(manifest.columns)


def test_mapping_conditions_fill_control(ravdess_csv, tmp_path):
    slots = {e: e for e in EMOTIONS}
    slots.update(neutral=None, fearful="afraid")
    manifest = build_manifest(emotion_config(ravdess_csv, conditions=slots),
                              system="indextts2", output_dir=tmp_path)
    by_condition = manifest.drop_duplicates("condition").set_index("condition")["control"]
    assert pd.isna(by_condition["neutral"])
    assert by_condition["fearful"] == "afraid"


def test_named_voices_get_formatted_descriptions(tmp_path):
    config = {"experiment": "emotion", "seed": 0, "anchor": "neutral",
              "sentences": SENTENCES, "speakers": ["Jon", "Lea"],
              "template": "{name} speaks {phrase}.",
              "conditions": {"neutral": "in a neutral tone", "sad": "in a sad tone"}}
    manifest = build_manifest(config, system="qwen3tts", output_dir=tmp_path)
    assert len(manifest) == 2 * 2 * 2
    assert "Jon speaks in a sad tone." in set(manifest["control"])
    assert manifest["clip_root"].isna().all()


def test_voice_descriptions_fill_the_template(tmp_path):
    config = {"experiment": "emotion", "seed": 0, "anchor": "neutral",
              "sentences": SENTENCES,
              "speakers": {"F25": "Female, 25 years old", "M55": "Male, 55 years old"},
              "template": "{voice}. Speak {phrase}.",
              "conditions": {"neutral": "in a neutral tone", "sad": "in a sad tone"}}
    manifest = build_manifest(config, system="qwen3tts", output_dir=tmp_path)
    assert len(manifest) == 2 * 2 * 2
    assert set(manifest["speaker_id"]) == {"F25", "M55"}
    row = manifest[(manifest["speaker_id"] == "F25") & (manifest["condition"] == "sad")
                   & (manifest["sentence_id"] == "kids")].iloc[0]
    assert row["control"] == "Female, 25 years old. Speak in a sad tone."
    assert row["output_path"] == "audio/F25/kids/sad.wav"


def test_effort_manifest_uses_complete_sets_and_avid_texts(avid_root, tmp_path):
    root, annotation = avid_root
    manifest = build_manifest(effort_config(root, annotation), system="cosyvoice3",
                              output_dir=tmp_path)
    # Complete sets: speaker 3 sentences 1 and 2, speaker 7 sentence 1 -> 3 sets.
    assert manifest["pair_id"].nunique() == 3 * 3
    assert len(manifest) == 3 * 3 * len(LEVELS)
    assert (manifest["clip_root"] == ".").all()
    assert (tmp_path / manifest.iloc[0]["condition_clip"]).exists()
    assert set(manifest["text"]) == {"Sentence number 4.", "Sentence number 5.",
                                     "Sentence number 6."}
    assert manifest.iloc[0]["output_path"] == "audio/avid_3/1/4/soft.wav"
    assert len(pd.read_csv(tmp_path / "skipped_prompt_sets.csv")) == 1
    assert pd.read_csv(tmp_path / "target_texts.csv")["n_variants"].tolist() == [1, 2, 1]
    # The prompts' own metadata rows, to measure the prompts later.
    prompts = pd.read_csv(tmp_path / "prompt_sets.csv")
    assert len(prompts) == 3 * len(LEVELS)
    assert {"signal_path", "start_s", "end_s", "channel", "clip"} <= set(prompts.columns)
    assert set(prompts["clip"]) == set(manifest["condition_clip"])


def test_effort_anchor_clip_is_the_normal_take(avid_root, tmp_path):
    root, annotation = avid_root
    manifest = build_manifest(effort_config(root, annotation), system="indextts2",
                              output_dir=tmp_path)
    assert manifest["anchor_clip"].str.endswith("/normal.wav").all()
    loud = manifest[manifest["condition"] == "loud"]
    assert loud["condition_clip"].str.endswith("/loud.wav").all()


def test_prompt_and_target_sentences_must_be_disjoint(avid_root, tmp_path):
    root, annotation = avid_root
    with pytest.raises(ValueError, match=r"\[2\] are both prompt and target"):
        build_manifest(effort_config(root, annotation, target_sentence_ids=[2, 4]),
                       system="cosyvoice3", output_dir=tmp_path)


def test_named_voices_effort_takes_avid_texts(avid_root, tmp_path):
    _, annotation = avid_root
    config = {"experiment": "effort", "seed": 0, "anchor": "normal",
              "speakers": ["Jon"], "template": "{name} speaks {phrase}.",
              "conditions": {"soft": "softly", "normal": "at a normal volume",
                             "loud": "loudly", "veryloud": "very loudly"},
              "avid": {"metadata_csv": str(annotation), "repetition": 1,
                       "target_sentence_ids": [4, 5]}}
    manifest = build_manifest(config, system="qwen3tts", output_dir=tmp_path)
    assert len(manifest) == 2 * 4
    assert manifest.iloc[0]["output_path"] == "audio/Jon/4/soft.wav"


def test_check_manifest_catches_a_missing_condition(ravdess_csv, tmp_path):
    manifest = build_manifest(emotion_config(ravdess_csv), system="cosyvoice3",
                              output_dir=tmp_path)
    with pytest.raises(ValueError, match="1 pair"):
        check_manifest(manifest.iloc[1:], EMOTIONS)


def test_check_manifest_catches_a_copied_prompt(ravdess_csv, tmp_path):
    manifest = build_manifest(emotion_config(ravdess_csv), system="cosyvoice3",
                              output_dir=tmp_path)
    manifest.loc[0, "condition_clip_text"] = manifest.loc[0, "text"]
    with pytest.raises(ValueError, match="prompt says the target sentence"):
        check_manifest(manifest, EMOTIONS)


def test_anchor_must_be_a_condition(ravdess_csv, tmp_path):
    with pytest.raises(ValueError, match="anchor 'calmness'"):
        build_manifest(emotion_config(ravdess_csv, anchor="calmness"), system="x",
                       output_dir=tmp_path)
