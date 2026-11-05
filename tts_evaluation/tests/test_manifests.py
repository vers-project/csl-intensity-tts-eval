"""Tests for the evaluation manifests."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from conftest import SENTENCES
from tts_evaluation import manifests
from speech_eval.io import load_manifest

RUN = "10-20-26/15-50-05"


def test_failed_rows_are_dropped_and_a_retry_counts_by_its_last_attempt(tts_root):
    rows = manifests.tts_rows("qwen3tts", "emotion", RUN, tts_root)

    # 2 jobs × 4 rows, minus job 1's failed one; job 0's failed-then-ok row is kept.
    assert len(rows) == 7
    assert "qwen3tts|emotion|F20|-|kids|0|angry" in set(rows["utt_id"])
    assert "qwen3tts|emotion|M30|-|kids|1|angry" not in set(rows["utt_id"])
    assert (rows["generation_status"] == "ok").all()


def test_tts_rows_carry_the_pair_and_a_path_relative_to_the_system_root(tts_root):
    rows = manifests.tts_rows("qwen3tts", "emotion", RUN, tts_root)

    assert rows["utt_id"].is_unique
    assert (rows["group_id"] == rows["pair_id"]).all()
    assert set(rows["corpus"]) == {"qwen3tts"}
    assert set(rows["source"]) == {"qwen3tts_emotion"}
    row = rows[rows["utt_id"] == "qwen3tts|emotion|M30|-|kids|1|neutral"].iloc[0]
    assert row["signal_path"] == f"emotion/{RUN}/1/audio/M30/kids/neutral.wav"
    assert (tts_root / row["signal_path"]).is_file()
    assert list(rows.columns[:len(manifests.ID_COLUMNS)]) == manifests.ID_COLUMNS


def test_a_manifest_row_missing_from_the_log_is_refused(tts_root):
    log = tts_root / "emotion" / RUN / "0" / "generation_log.csv"
    table = pd.read_csv(log)
    table[table["condition"] != "neutral"].to_csv(log, index=False)

    with pytest.raises(ValueError, match="no line in generation_log"):
        manifests.tts_rows("qwen3tts", "emotion", RUN, tts_root)


def test_ravdess_rows_use_the_generation_sentence_ids(ravdess_csv):
    _, csv = ravdess_csv
    rows = manifests.ravdess_rows(csv, SENTENCES)

    assert len(rows) == 12                                  # 2 actors × 3 takes × 2 statements
    assert set(rows["sentence_id"]) == {"kids", "dogs"}
    assert set(rows["speaker_id"]) == {"actor_01", "actor_02"}
    # The parallel unit: one group per actor × statement × repetition, 3 takes each.
    assert rows.groupby("group_id").size().tolist() == [3] * 4
    assert rows.loc[rows["is_anchor"], "condition"].unique().tolist() == ["neutral"]


def test_an_unnamed_ravdess_statement_is_refused(ravdess_csv):
    _, csv = ravdess_csv
    with pytest.raises(ValueError, match="no id"):
        manifests.ravdess_rows(csv, {"kids": SENTENCES["kids"]})


def test_avid_rows_keep_the_split_and_its_calibration(avid_csv):
    _, csv = avid_csv
    rows = manifests.avid_rows(csv)

    assert len(rows) == 3                                   # the train and VIC rows are out
    assert rows["utt_id"].tolist() == ["Spk21/session.wav|0", "Spk21/session.wav|1",
                                       "Spk21/session.wav|2"]
    assert rows["group_id"].iloc[0] == rows["group_id"].iloc[1] == "avid_21|1|1|4"
    assert pd.isna(rows["group_id"].iloc[2])                # unannotated: no group
    assert rows["is_anchor"].tolist() == [False, True, False]
    assert (rows["channel"] == 0).all()
    assert {"calibration_rms", "distance_m", "level"} <= set(rows.columns)


def test_every_source_carries_sex_as_f_or_m(tts_root, ravdess_csv, avid_csv):
    ravdess = manifests.ravdess_rows(ravdess_csv[1], SENTENCES)
    avid = manifests.avid_rows(avid_csv[1])
    sex_of = manifests.speaker_sex(ravdess, avid)
    assert sex_of == {"actor_01": "M", "actor_02": "F", "avid_21": "F"}

    # Qwen3 voice ids name their sex; a natural speaker's entry wins over the id rule.
    qwen = manifests.tts_rows("qwen3tts", "emotion", RUN, tts_root, sex_of)
    assert dict(zip(qwen["speaker_id"], qwen["sex"])) == {"F20": "F", "M30": "M"}
    cloned = manifests.tts_sex(pd.Series(["actor_01", "avid_21", "F20"]),
                               sex_of | {"F20": "M"})
    assert cloned.tolist() == ["M", "F", "M"]


def test_a_tts_speaker_without_a_sex_is_refused():
    with pytest.raises(ValueError, match="No sex for TTS speaker 'actor_09'"):
        manifests.tts_sex(pd.Series(["actor_09"]), {"actor_01": "M"})


def test_a_speaker_with_two_sexes_is_refused():
    rows = pd.DataFrame({"speaker_id": ["actor_01", "actor_01"], "sex": ["M", "F"]})
    with pytest.raises(ValueError, match="more than one sex"):
        manifests.speaker_sex(rows)


def test_the_trim_moves_the_span_and_keeps_paths_relative(tts_root, stub_vad):
    rows = manifests.tts_rows("qwen3tts", "emotion", RUN, tts_root)
    trimmed = manifests.trim(rows, {"qwen3tts": tts_root}, progress=False)

    assert trimmed["signal_path"].tolist() == rows["signal_path"].tolist()
    assert trimmed["start_s"].tolist() == pytest.approx([0.25] * len(rows))
    assert trimmed["end_s"].tolist() == pytest.approx([1.5] * len(rows))
    assert trimmed["trimmed_tail_s"].tolist() == pytest.approx([0.5] * len(rows))
    assert (trimmed["vad_status"] == "ok").all()


def test_the_avid_trim_is_relative_to_each_segment(avid_csv, stub_vad):
    root, csv = avid_csv
    trimmed = manifests.trim(manifests.avid_rows(csv), {"AVID": root}, progress=False)

    assert trimmed["start_s"].tolist() == pytest.approx([0.25, 2.25, 4.25])
    assert trimmed["untrimmed_start_s"].tolist() == pytest.approx([0.0, 2.0, 4.0])


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_eval_manifests.py"
    spec = importlib.util.spec_from_file_location("build_eval_manifests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_writes_manifests_speech_eval_can_load(tmp_path, monkeypatch, tts_root,
                                                         ravdess_csv, avid_csv, stub_vad):
    ravdess_root, ravdess_meta = ravdess_csv
    avid_root, avid_meta = avid_csv
    roots = {"qwen3tts": str(tts_root), "RAVDESS": str(ravdess_root), "AVID": str(avid_root)}
    config = {
        "dataset_roots": roots,
        "tts": {"qwen3tts": {"emotion": RUN}},
        "ravdess": {"metadata_csv": str(ravdess_meta), "sentences": SENTENCES},
        "avid": {"metadata_csv": str(avid_meta), "corpus": "AVID", "split": "test"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "out"

    monkeypatch.setattr(sys, "argv", ["build_eval_manifests.py", "--config",
                                      str(config_path), "--output-dir", str(out)])
    _load_script().main()

    summary = pd.read_csv(out / "summary.csv")
    assert sorted(summary["source"]) == ["avid", "qwen3tts_emotion", "ravdess"]
    assert (summary["no_speech"] == 0).all()
    for source in summary["source"]:
        manifest = load_manifest(out / "manifests" / f"{source}.csv", dataset_roots=roots)
        assert manifest["signal_path"].map(lambda p: Path(p).is_file()).all()
        assert set(manifest["sex"]) <= {"F", "M"}
