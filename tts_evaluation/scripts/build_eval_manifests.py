"""Build one VAD-trimmed evaluation manifest per source.

Writes ``<output_dir>/manifests/<source>.csv`` for every TTS run and natural corpus in the
config, and ``summary.csv`` (rows, rows without speech, median trimmed head / tail).  A
source whose manifest already exists is skipped, so a killed job is simply resubmitted.

Usage
-----
    uv run --extra cpu launch_experiment --script scripts/build_eval_manifests.py \\
        --config configs/manifests.yaml
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from experiment_launcher import parse_args

from tts_evaluation import manifests


def source_builders(config: dict) -> dict:
    """``{source: zero-argument builder}`` in the order the sources are processed."""
    roots = config["dataset_roots"]
    natural = {}
    if config.get("ravdess"):
        natural["ravdess"] = partial(manifests.ravdess_rows,
                                     config["ravdess"]["metadata_csv"],
                                     config["ravdess"]["sentences"])
    if config.get("avid"):
        natural["avid"] = partial(manifests.avid_rows, **config["avid"])

    builders = {}
    tts = config.get("tts") or {}
    if tts:
        # A cloned voice has the sex of its natural speaker (metadata only, no trim).
        sex_of = manifests.speaker_sex(*(build() for build in natural.values()))
        for system, runs in tts.items():
            for experiment, run_folder in runs.items():
                builders[f"{system}_{experiment}"] = partial(
                    manifests.tts_rows, system, experiment, run_folder, roots[system],
                    sex_of)
    return builders | natural


@parse_args
def main(config: dict, output_dir: Path) -> None:
    out = Path(output_dir) / "manifests"
    out.mkdir(parents=True, exist_ok=True)

    for source, build in source_builders(config).items():
        path = out / f"{source}.csv"
        if path.exists():
            print(f"[{source}] {path} exists, skipped.")
            continue
        rows = build()
        manifests.check_unique_utt_ids(rows, source)
        print(f"[{source}] {len(rows)} rows, trimming.")
        rows = manifests.trim(rows, config["dataset_roots"], config.get("vad"))
        rows.to_csv(path, index=False)

    summary = pd.DataFrame([manifests.summarise(pd.read_csv(p, low_memory=False))
                            for p in sorted(out.glob("*.csv"))])
    summary.to_csv(Path(output_dir) / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
