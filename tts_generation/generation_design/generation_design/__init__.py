"""Controlled generation design shared by the TTS systems.

Every (speaker, sentence, seed) is generated once under each condition, so the samples of
such a group, identified by ``pair_id``, differ only in the control input.  This package decides what to generate and runs the
generation loop; each TTS system's ``generate_parallel.py`` only supplies the function that
turns one manifest row into audio.

    design.build_manifest   config -> manifest.csv (+ the AVID prompts cut to disk)
    run.run_manifest        manifest -> audio/*.wav + generation_log.csv
"""
from generation_design.design import build_manifest, check_manifest
from generation_design.run import run_manifest, seed_everything

__all__ = ["build_manifest", "check_manifest", "run_manifest", "seed_everything"]
