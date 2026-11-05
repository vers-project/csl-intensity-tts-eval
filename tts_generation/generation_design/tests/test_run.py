import numpy as np
import pandas as pd
import soundfile as sf

from conftest import DOGS, KIDS
from generation_design.design import build_manifest
from generation_design.run import run_manifest

CONFIG = {"experiment": "emotion", "seed": 5, "anchor": "neutral",
          "sentences": {"kids": KIDS, "dogs": DOGS}, "speakers": ["Jon"],
          "conditions": ["neutral", "sad", "angry"]}


def noise_generator(row):
    """Random draws only: equal outputs mean the RNG was re-seeded before each call."""
    return np.random.standard_normal(100) * 2.0, 16000


def test_rows_are_seeded_and_written_as_float32(tmp_path):
    manifest = build_manifest(CONFIG, system="qwen3tts", output_dir=tmp_path)
    log = run_manifest(manifest, noise_generator, tmp_path)
    assert (log["status"] == "ok").all()
    first = tmp_path / manifest.iloc[0]["output_path"]
    assert sf.info(str(first)).subtype == "FLOAT"
    audios = [sf.read(str(tmp_path / p), dtype="float32")[0] for p in manifest["output_path"]]
    for audio in audios[1:]:
        np.testing.assert_array_equal(audio, audios[0])
    assert np.abs(audios[0]).max() > 1.0          # no clipping or rescaling


def test_failure_is_logged_and_the_loop_continues(tmp_path):
    manifest = build_manifest(CONFIG, system="qwen3tts", output_dir=tmp_path)

    def flaky(row):
        if row["condition"] == "sad":
            raise RuntimeError("model blew up")
        return np.zeros(10), 16000

    log = run_manifest(manifest, flaky, tmp_path)
    assert log["status"].value_counts().to_dict() == {"ok": 4, "failed": 2}
    assert log.loc[log["status"] == "failed", "error"].str.contains("model blew up").all()


def test_relaunch_skips_existing_and_retries_failures(tmp_path):
    manifest = build_manifest(CONFIG, system="qwen3tts", output_dir=tmp_path)

    def fails_on_sad(row):
        if row["condition"] == "sad":
            raise RuntimeError("x")
        return np.zeros(10), 16000

    run_manifest(manifest, fails_on_sad, tmp_path)
    calls = []

    def recording(row):
        calls.append(row["condition"])
        return np.zeros(10), 16000

    log = run_manifest(manifest, recording, tmp_path)
    assert calls == ["sad", "sad"]
    assert log["status"].value_counts().to_dict() == {"exists": 4, "ok": 2}
    history = pd.read_csv(tmp_path / "generation_log.csv")
    assert len(history) == 12                     # both runs appended, header written once


def test_bad_waveforms_fail_the_row(tmp_path):
    manifest = build_manifest(CONFIG, system="qwen3tts", output_dir=tmp_path)
    log = run_manifest(manifest, lambda row: (np.full(10, np.nan), 16000), tmp_path)
    assert (log["status"] == "failed").all()
    assert log["error"].str.contains("NaN").all()


def test_generate_receives_absolute_clip_paths(ravdess_csv, tmp_path):
    config = {"experiment": "emotion", "seed": 0, "anchor": "neutral",
              "sentences": {"kids": KIDS, "dogs": DOGS}, "conditions": ["neutral", "sad"],
              "ravdess": {"metadata_csv": str(ravdess_csv), "root": "/data/RAVDESS",
                          "actors": [1]}}
    manifest = build_manifest(config, system="cosyvoice3", output_dir=tmp_path)
    seen = []

    def capture(row):
        seen.append(row)
        return np.zeros(10), 16000

    run_manifest(manifest, capture, tmp_path)
    assert seen[0]["condition_clip"].startswith("/data/RAVDESS/Actor_01/")
    assert seen[0]["output_path"].startswith(str(tmp_path))
