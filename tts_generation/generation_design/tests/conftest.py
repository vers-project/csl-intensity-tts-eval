"""Tiny synthetic RAVDESS and AVID tables (and AVID session recordings) for the tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

KIDS = "Kids are talking by the door"
DOGS = "Dogs are sitting by the door"
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
LEVELS = ["soft", "normal", "loud", "veryloud"]
SR = 8000


@pytest.fixture
def ravdess_csv(tmp_path):
    rows = []
    for actor in (1, 2):
        for emotion in EMOTIONS:
            intensities = ["normal"] if emotion == "neutral" else ["normal", "strong"]
            for intensity in intensities:
                for statement in (KIDS, DOGS):
                    for repetition in (1, 2):
                        tag = f"{actor}-{emotion}-{intensity}-{statement[:4]}-{repetition}"
                        rows.append({"signal_path": f"Actor_{actor:02d}/{tag}.wav",
                                     "actor": actor, "emotion": emotion,
                                     "intensity": intensity, "statement": statement,
                                     "repetition": repetition})
    path = tmp_path / "ravdess.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def avid_root(tmp_path):
    """Two speakers, one session file each: channel 0 = +0.5, channel 1 (EGG) = -0.5.

    The table has the layout of the P_φ prediction metadata with the annotation merged:
    every corpus in one file, ids read back as floats, a VIC row carrying its own `level`
    and an unannotated AVID train row, both of which must be ignored.

    Speaker 3 reads sentences 1-6 at the four levels.  Speaker 7 misses the `loud` take
    of sentence 2, and has an excluded `soft` take of sentence 1 next to a kept one.
    Sentence 5 is misread once by speaker 7.
    """
    root = tmp_path / "AVID"
    root.mkdir()
    rows = []
    for speaker in (3, 7):
        signal_path = f"Spk{speaker}/session.wav"
        (root / f"Spk{speaker}").mkdir()
        t = 0.0
        index = 0
        for sentence in range(1, 7):
            for level in LEVELS:
                if speaker == 7 and sentence == 2 and level == "loud":
                    continue
                text = f"Sentence number {sentence}."
                if speaker == 7 and sentence == 5 and level == "soft":
                    text = "Sentence numbr five."
                rows.append({"signal_path": signal_path, "segment_index": index,
                             "start_s": t, "end_s": t + 0.5, "text": text,
                             "subject_id": speaker, "channel": 0, "level": level,
                             "sentence_id": sentence, "repetition": 1, "excluded": False,
                             "session_index": 1})
                index += 1
                t += 0.5
            if speaker == 7 and sentence == 1:
                rows.append({"signal_path": signal_path, "segment_index": index,
                             "start_s": t, "end_s": t + 0.5, "text": "Sentence number 1.",
                             "subject_id": speaker, "channel": 0, "level": "soft",
                             "sentence_id": 1, "repetition": 1, "excluded": True,
                             "session_index": 1})
                index += 1
                t += 0.5
        # A second repetition that must be ignored.
        rows.append({"signal_path": signal_path, "segment_index": index, "start_s": 0.0,
                     "end_s": 0.5, "text": "Sentence number 1.", "subject_id": speaker,
                     "channel": 0, "level": "soft", "sentence_id": 1, "repetition": 2,
                     "excluded": False, "session_index": 1})
        n = int((t + 0.5) * SR)
        audio = np.stack([np.full(n, 0.5), np.full(n, -0.5)], axis=1).astype(np.float32)
        sf.write(str(root / signal_path), audio, SR, subtype="FLOAT")
    table = pd.DataFrame(rows).assign(corpus="AVID", split="test")
    others = pd.DataFrame([
        {"corpus": "VIC", "split": "test", "signal_path": "vic/a.wav", "segment_index": 0,
         "start_s": 0.0, "end_s": 1.0, "text": "Sentence number 1.", "level": "loud",
         "sentence_id": 1, "repetition": 1, "excluded": False},
        {"corpus": "AVID", "split": "train", "signal_path": "Spk1/session.wav",
         "segment_index": 0, "start_s": 0.0, "end_s": 1.0, "text": "Sentence number 1.",
         "subject_id": 1},
    ])
    annotation = root / "metadata.csv"
    # The concat turns subject_id, channel and session_index into floats, as in the real file.
    pd.concat([table, others], ignore_index=True).to_csv(annotation, index=False)
    return root, annotation
