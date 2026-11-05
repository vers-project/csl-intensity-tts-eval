"""Absolute attribute distributions per condition, every system and natural corpus.

Reads the attribute table (``attributes.csv``). Each ``joins`` entry adds columns of another
per-source run's ``predictions/utterances.csv`` as ``<name>.<key>`` (job ↔ source from its
``sweep_lookup.csv``). No difference is taken; see ``tts_evaluation.distributions``.
Writes ``<output_dir>/distributions.csv``: per (``by`` columns, attribute), n, n_speakers,
mean, sd and quantiles; plus, for each ``pool`` column, the same rows pooled over it, with
that column set to ``all``.

Usage
-----
    uv run --extra cpu launch_experiment --script scripts/attribute_distributions.py \\
        --config configs/distributions.yaml
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment_launcher import parse_args

from tts_evaluation import attributes, distributions


@parse_args
def main(config: dict, output_dir: Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(config["attributes_csv"], low_memory=False)
    joins = {name: {"jobs": attributes.run_jobs(join["run"]), "columns": join["columns"]}
             for name, join in (config.get("joins") or {}).items()}

    described = []
    for source, spec in config["sources"].items():
        frame = table[table["source"] == source]
        if (spec or {}).get("query"):
            frame = frame.query(spec["query"])
        if frame.empty:
            raise ValueError(f"[{source}] no row in {config['attributes_csv']}.")
        for name, join in joins.items():
            frame = distributions.join_utterances(
                frame, join["jobs"][source] / "predictions" / "utterances.csv", name,
                join["columns"])
        described.append(distributions.describe(frame, config["attributes"], config["by"],
                                                config["quantiles"]))
        # Quantiles do not merge: a pooled row is computed, labelled "all".
        for column in config.get("pool", []):
            by = [c for c in config["by"] if c != column]
            pooled = distributions.describe(frame, config["attributes"], by,
                                            config["quantiles"])
            described.append(pooled.assign(**{column: "all"}))
        print(f"[{source}] {len(frame)} rows.")

    summary = pd.concat(described, ignore_index=True)
    summary.to_csv(out / "distributions.csv", index=False)

    headline = summary[summary["attribute"] == "intensity"]
    print(headline.to_string(index=False, float_format="%.1f"))


if __name__ == "__main__":
    main()
