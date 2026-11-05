"""Build the attribute table and the failure screen.

Joins each source's evaluation manifest, the predictor's output and the speech-eval metrics,
and flags the samples with no speech (see ``tts_evaluation.attributes``). No audio is read.
Writes to ``<output_dir>/``:

    attributes.csv       one row per utterance, all sources
    screen_summary.csv   rows and failures per reason, per source × condition
    coverage.csv         NaN count per source for each analysis attribute

Usage
-----
    uv run --extra cpu launch_experiment --script scripts/build_attribute_table.py \\
        --config configs/attributes.yaml
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment_launcher import parse_args

from tts_evaluation import attributes


@parse_args
def main(config: dict, output_dir: Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = config["runs"]
    pphi_jobs = attributes.run_jobs(runs["pphi"])
    metrics_jobs = attributes.run_jobs(runs["metrics"])

    frames = []
    for source in config["sources"]:
        frame = attributes.load_source(
            Path(runs["manifests"]) / f"{source}.csv", pphi_jobs[source], metrics_jobs[source])
        frame = attributes.drop_excluded(frame, source)
        frames.append(attributes.flag_failures(frame))
        print(f"[{source}] {len(frame)} rows.")

    table = pd.concat(frames, ignore_index=True)
    table.to_csv(out / "attributes.csv", index=False)
    summary = attributes.failure_summary(table)
    summary.to_csv(out / "screen_summary.csv", index=False)
    coverage = attributes.coverage(table, config["attributes"])
    coverage.to_csv(out / "coverage.csv", index=False)

    print(summary.to_string(index=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
