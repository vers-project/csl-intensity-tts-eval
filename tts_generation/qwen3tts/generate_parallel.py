"""Generate the Qwen3-TTS VoiceDesign samples of one experiment (emotion or effort).

Control = text only.  Each manifest row's ``control`` is the full instruction (the config's
template with the voice description and the condition phrase filled in); it sets both the
voice and the style.

The manifest, seeding and saving live in ``generation_design``; this script only turns a row
into audio.

Usage
-----
  uv run --extra cu128 launch_experiment --script generate_parallel.py --config configs/emotion.yaml
"""
from __future__ import annotations

from pathlib import Path

import torch
from experiment_launcher import parse_args
from generation_design import build_manifest, run_manifest


def make_generate(model, language: str):
    def generate(row: dict):
        wavs, sample_rate = model.generate_voice_design(
            text=row["text"], language=language, instruct=row["control"])
        return wavs[0], sample_rate
    return generate


@parse_args
def main(config: dict, output_dir: Path) -> None:
    manifest = build_manifest(config, system="qwen3tts", output_dir=output_dir)
    if config.get("dry_run", False):
        print("Dry run: manifest only.")
        return

    from qwen_tts import Qwen3TTSModel

    # sdpa attention: flash-attn is not required.
    model = Qwen3TTSModel.from_pretrained(config["model_dir"], device_map="cuda:0",
                                          dtype=torch.bfloat16, attn_implementation="sdpa")
    print(f"Loaded Qwen3-TTS ({model.model.tts_model_type}) from {config['model_dir']}, "
          f"sampling defaults {model.generate_defaults}")

    run_manifest(manifest, make_generate(model, config.get("language", "English")), output_dir)


if __name__ == "__main__":
    main()
