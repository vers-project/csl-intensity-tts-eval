"""Write the numbers quoted in the paper text as LaTeX macros (tts_evaluation.numbers).

Reads the distributions run of record (and, for the failure rates, the attribute table named in
that run's config.yaml, filtered by the run's own source queries) and writes ``paper_numbers_tex`` (``\\input`` in the
preamble of main.tex).  Prints every macro and the signed system − natural differences, so
the words of the text ("louder", "softer") can be checked against the signs.

Usage
-----
    uv run --extra cpu launch_experiment --script scripts/make_paper_numbers.py \\
        --config configs/numbers.yaml
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from experiment_launcher import parse_args

from tts_evaluation import attributes, distributions, numbers


@parse_args
def main(config: dict, output_dir: Path) -> None:
    source = Path(config["distributions"]) / "distributions.csv"
    summary = pd.read_csv(source)
    conditions = {e: list(c) for e, c in config["conditions"].items()}
    differences = {name: (pair["a"], pair["b"]) for name, pair in config["differences"].items()}
    values = numbers.paper_numbers(summary, conditions, differences)
    values |= numbers.rms_numbers(summary, conditions, config["rms_attribute"])

    # Failures: the attribute table, filtered by the distributions run's own source queries.
    run_config = yaml.safe_load((Path(config["distributions"]) / "config.yaml").read_text())
    queries = {src: (spec or {}).get("query") for src, spec in run_config["sources"].items()}
    attributes_csv = run_config["attributes_csv"]
    table = pd.read_csv(attributes_csv, low_memory=False)
    counts = numbers.failure_counts(table, queries)
    values |= numbers.failure_numbers(counts, config["failure_exclude"])

    values |= numbers.alpha_agreement(summary, conditions, config["alpha_attribute"])

    # Calibrated natural speech: one row per sample, with the alpha regression joined as the
    # distributions run joined it.  Its predictor medians must be the ones of Figure 1.
    spec = config["calibrated"]
    join = run_config["joins"][spec["join"]]
    rows = distributions.join_utterances(
        table[table["source"] == spec["source"]],
        attributes.run_jobs(join["run"])[spec["source"]] / "predictions" / "utterances.csv",
        spec["join"], join["columns"])
    effort = conditions["effort"]
    per_sample = rows.groupby("condition")[spec["predictor"]].median()
    figure = numbers.medians(summary).loc["effort"][numbers.NATURAL]
    if any(numbers.round1(per_sample[c]) != numbers.round1(figure[c]) for c in effort):
        raise ValueError(f"{spec['source']} rows do not give the medians of Figure 1")
    values |= numbers.calibrated_numbers(rows, effort, spec["label"], spec["predictor"],
                                         spec["alpha"])

    tex = numbers.latex(values, f"{source}; {attributes_csv}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partII_numbers.tex").write_text(tex)
    target = Path(config["paper_numbers_tex"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(tex)

    print(tex)
    print(numbers.signed_listing(summary, conditions).to_string(index=False))
    print(counts.assign(rate=100 * counts["failed"] / counts["total"]).round(1).to_string())
    print(f"Written → {target}")


if __name__ == "__main__":
    main()
