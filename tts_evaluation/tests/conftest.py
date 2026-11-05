"""A synthetic generation-output layout, tiny RAVDESS / AVID tables, and a stubbed Silero."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

import audio_utils.data.vad as vad_module

SR = 16000
KIDS = "Kids are talking by the door"
DOGS = "Dogs are sitting by the door"
SENTENCES = {"kids": KIDS, "dogs": DOGS}


def _wav(path, seconds=2.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(seconds * SR), dtype="float32"), SR)


@pytest.fixture
def tts_root(tmp_path):
    """``qwen3tts/emotion/10-20-26/15-50-05/{0,1}``: 2 jobs (seeds) × 2 pairs × 2 conditions.

    Job 0 logs one row as failed and then ok (a resubmission: the last attempt counts);
    job 1 logs one row as failed only.
    """
    root = tmp_path / "qwen3tts"
    for job, seed in ((0, 0), (1, 1)):
        job_dir = root / "emotion" / "10-20-26" / "15-50-05" / str(job)
        manifest, log = [], []
        for speaker in ("F20", "M30"):
            pair_id = f"{speaker}|-|kids|{seed}"
            for condition in ("neutral", "angry"):
                output_path = f"audio/{speaker}/kids/{condition}.wav"
                _wav(job_dir / output_path)
                manifest.append({
                    "pair_id": pair_id, "system": "qwen3tts", "experiment": "emotion",
                    "seed": seed, "speaker_id": speaker, "prompt_sentence_id": np.nan,
                    "sentence_id": "kids", "text": KIDS, "condition": condition,
                    "is_anchor": condition == "neutral", "control": f"{speaker}. {condition}",
                    "output_path": output_path})
                log.append({"pair_id": pair_id, "condition": condition,
                            "output_path": output_path, "status": "ok"})
        if job == 0:
            log.insert(0, {"pair_id": "F20|-|kids|0", "condition": "angry",
                           "output_path": "audio/F20/kids/angry.wav", "status": "failed"})
        else:
            log[-1]["status"] = "failed"          # M30 angry, seed 1
        pd.DataFrame(manifest).to_csv(job_dir / "manifest.csv", index=False)
        pd.DataFrame(log).to_csv(job_dir / "generation_log.csv", index=False)
    # A stale folder next to the job folders, without a manifest: ignored.
    (root / "emotion" / "10-20-26" / "15-50-05" / ".submitit").mkdir()
    return root


@pytest.fixture
def ravdess_csv(tmp_path):
    root = tmp_path / "RAVDESS"
    rows = []
    for actor in (1, 2):
        for emotion, intensities in (("neutral", ["normal"]), ("angry", ["normal", "strong"])):
            for intensity in intensities:
                for statement in (KIDS, DOGS):
                    path = f"Actor_{actor:02d}/{emotion}-{intensity}-{statement[:4]}.wav"
                    _wav(root / path)
                    rows.append({"signal_path": path, "actor": actor,
                                 "gender": "male" if actor % 2 else "female",
                                 "emotion": emotion,
                                 "intensity": intensity, "statement": statement,
                                 "repetition": 1})
    csv = tmp_path / "ravdess.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return root, csv


@pytest.fixture
def avid_csv(tmp_path):
    """One AVID session (2 annotated takes + 1 unannotated), a train row and a VIC row.

    The concat turns ids into floats, as in the real all-corpora table.
    """
    root = tmp_path / "AVID"
    _wav(root / "Spk21/session.wav", seconds=6.0)
    annotated = [
        {"signal_path": "Spk21/session.wav", "segment_index": i, "start_s": 2.0 * i,
         "end_s": 2.0 * i + 2.0, "subject_id": 21, "sex": "F", "level": level, "sentence_id": 4,
         "repetition": 1, "session_index": 1, "excluded": False, "text": "Hi.",
         "calibration_rms": 0.04, "distance_m": 0.05}
        for i, level in enumerate(("soft", "normal"))]
    unannotated = {"signal_path": "Spk21/session.wav", "segment_index": 2, "start_s": 4.0,
                   "end_s": 6.0, "subject_id": 21, "sex": "F", "text": "Bye.",
                   "calibration_rms": 0.04, "distance_m": 0.05}
    table = pd.DataFrame(annotated + [unannotated]).assign(corpus="AVID", split="test")
    others = pd.DataFrame([
        {"corpus": "AVID", "split": "train", "signal_path": "Spk1/s.wav", "segment_index": 0,
         "subject_id": 1},
        {"corpus": "VIC", "split": "test", "signal_path": "vic/a.wav", "segment_index": 0,
         "level": "loud"},
    ])
    csv = tmp_path / "avid.csv"
    pd.concat([table, others], ignore_index=True).to_csv(csv, index=False)
    return root, csv


class SileroStub:
    def reset_states(self):
        pass

    def audio_forward(self, x, sr):
        window = 512 if sr == 16000 else 256
        return torch.zeros(x.shape[0], -(-x.shape[-1] // window))


@pytest.fixture
def stub_vad(monkeypatch):
    """Silero reports one island, 0.25 s → 1.5 s of whatever region it is given."""
    import silero_vad

    def fake(audio, model, sampling_rate, **kwargs):
        model.reset_states()
        window = 512 if sampling_rate == 16000 else 256
        for start in range(0, len(audio), window):
            model(audio[start:start + window], sampling_rate)
        return [{"start": round(0.25 * sampling_rate), "end": round(1.5 * sampling_rate)}]

    monkeypatch.setattr(silero_vad, "get_speech_timestamps", fake)
    monkeypatch.setattr(vad_module, "load_vad_model", lambda onnx=False: SileroStub())
