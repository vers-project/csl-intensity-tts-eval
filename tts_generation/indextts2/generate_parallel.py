"""Generate the IndexTTS2 samples of one experiment (emotion or effort).

The speaker prompt is always the row's ``anchor_clip`` (RAVDESS neutral take / AVID normal
take), so it is identical across the conditions of a group.  The control depends on the
experiment:

  emotion  one-hot emotion vector, ``emo_vector_weight`` on the slot named by ``control``;
           the anchor row (``control`` empty) gets no vector.
  effort   emotion audio prompt = the row's ``condition_clip`` (the same speaker's AVID take
           at that level), weighted by ``emo_alpha``.

The manifest, seeding and saving live in ``generation_design``; this script only turns a row
into audio.

Usage
-----
  uv run launch_experiment --script generate_parallel.py --config configs/emotion.yaml
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from experiment_launcher import parse_args
from generation_design import build_manifest, run_manifest

# Emotion-vector slot order of IndexTTS2.
SLOTS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


def emotion_vector(slot, weight: float) -> list[float] | None:
    if pd.isna(slot):
        return None
    vector = [0.0] * len(SLOTS)
    vector[SLOTS.index(slot)] = weight
    return vector


def load_model(model_config: dict):
    from indextts.infer_v2 import IndexTTS2

    model_dir = model_config["dir"]
    return IndexTTS2(
        cfg_path=str(Path(model_dir) / "config.yaml"),
        model_dir=model_dir,
        use_fp16=model_config.get("fp16", False),
        use_cuda_kernel=model_config.get("cuda_kernel", False),
        use_deepspeed=model_config.get("deepspeed", False),
    )


def make_generate(model, config: dict):
    experiment = config["experiment"]
    emo_alpha = config.get("emo_alpha", 1.0)
    use_random = config.get("use_random", False)

    def generate(row: dict):
        if experiment == "emotion":
            control = {"emo_vector": emotion_vector(row["control"], config["emo_vector_weight"])}
        else:
            control = {"emo_audio_prompt": row["condition_clip"]}
        # output_path=None returns (sample_rate, int16 array of shape (T, 1)).
        result = model.infer(spk_audio_prompt=row["anchor_clip"], text=row["text"],
                             output_path=None, emo_alpha=emo_alpha, use_random=use_random,
                             **control)
        if result is None:
            raise RuntimeError("no audio generated")
        sample_rate, audio = result
        return np.asarray(audio, dtype=np.float32) / 32768.0, sample_rate

    return generate


@parse_args
def main(config: dict, output_dir: Path) -> None:
    manifest = build_manifest(config, system="indextts2", output_dir=output_dir)
    unknown = sorted(set(manifest["control"].dropna()) - set(SLOTS))
    if config["experiment"] == "emotion" and unknown:
        raise ValueError(f"Unknown IndexTTS2 slot(s) {unknown}; valid: {SLOTS}")
    if config.get("dry_run", False):
        print("Dry run: manifest only.")
        return

    model = load_model(config["model"])
    run_manifest(manifest, make_generate(model, config), output_dir)


if __name__ == "__main__":
    main()
