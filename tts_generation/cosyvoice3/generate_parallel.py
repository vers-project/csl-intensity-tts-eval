"""Generate the CosyVoice 3 samples of one experiment (emotion or effort), zero-shot.

Control = acoustic prompt: each manifest row is synthesised from its ``condition_clip``
(RAVDESS emotion take or AVID effort take) with that clip's transcript as prompt text.
The manifest, seeding and saving live in ``generation_design``; this script only turns a row
into audio.

The output needs no trimming: every flow model drops the prompt frames itself
(``feat[:, :, mel_len1:]`` in ``cosyvoice/flow/flow.py``).

Usage
-----
  uv run launch_experiment --script generate_parallel.py --config configs/emotion.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

COSYVOICE_DIR = Path(__file__).parent / "CosyVoice"
sys.path[:0] = [str(COSYVOICE_DIR), str(COSYVOICE_DIR / "third_party" / "Matcha-TTS")]

import torch
from experiment_launcher import parse_args
from generation_design import build_manifest, run_manifest


def make_generate(model, system_prompt: str):
    def generate(row: dict):
        chunks = list(model.inference_zero_shot(
            # The wetext English normaliser spells out runs of capitals letter by letter.
            tts_text=row["text"].lower(),
            prompt_text=system_prompt + row["condition_clip_text"],
            prompt_wav=row["condition_clip"],
            stream=False,
        ))
        if not chunks:
            raise RuntimeError("no audio generated")
        return torch.cat([c["tts_speech"] for c in chunks], dim=-1), model.sample_rate
    return generate


@parse_args
def main(config: dict, output_dir: Path) -> None:
    manifest = build_manifest(config, system="cosyvoice3", output_dir=output_dir)
    if config.get("dry_run", False):
        print("Dry run: manifest only.")
        return

    from cosyvoice.cli.cosyvoice import CosyVoice3
    model = CosyVoice3(config["model_dir"], fp16=config.get("fp16", False))
    print(f"Loaded CosyVoice3, {model.sample_rate} Hz")

    run_manifest(manifest, make_generate(model, config.get("system_prompt") or ""), output_dir)


if __name__ == "__main__":
    main()
