"""Write the intensity estimation results table (LaTeX) from the predictor grid.

Usage
-----
    python make_results_table.py \
        --experiments_dir <grid> --alpha_baseline <dir> --hammarberg_baseline <dir> \
        --output results_table.tex

``<grid>`` is the output folder of configs/train_predictor_grid.yaml (one subfolder per
job); the two baseline folders are the ``alpha`` and ``hammarberg`` refits of
configs/extract_window_tilt.yaml.

The runs are gathered with ``collect_experiment_results.py`` (same directory): Lightning's
test metrics, aggregated over seeds.  The table has one block per metric (RMSE, bias),
one row per test corpus × feature extractor, and one column per time scale (frame,
each sliding window, utterance).  Each cell is the mean ± sample SD over the seeds.

Each corpus block opens with the tilt-only baselines, one row per tilt measure of
``TILT_FEATURES``: a ``metrics.csv`` of vocal-intensity-prediction's
``scripts/fit_tilt_baseline.py`` (or of ``scripts/extract_window_tilt.py``) whose ``feature`` column must name that measure,
degree ``TILT_DEGREE``.  They are deterministic, so their cells have no SD.

Every expected run, corpus, extractor and scale must be present, with ``N_SEEDS`` runs
per extractor: anything missing stops the script instead of leaving a blank cell.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from collect_experiment_results import aggregate_replicates, collect, to_table, to_tidy

N_SEEDS = 3
EXTRACTORS = {"melspectrogram": "Log-mel", "wavlm": "WavLM Base+", "wav2vec2": "wav2vec~2.0 XLS-R 300M"}
CORPORA = {"AVID": "AVID", "VIC": "VIC", "FLombard": "FLombard"}
# metric-name prefix of each time scale → column header
SCALES = {
    "": r"Frame (\SI{25}{\milli\second})",
    "win_0.26s/": r"\SI{0.26}{\second}",
    "win_0.5s/": r"\SI{0.5}{\second}",
    "win_1s/": r"\SI{1}{\second}",
    "leq/": "Utterance",
}
METRICS = {"rmse_db": ("RMSE", False), "me_db": ("Bias", True)}   # (label, signed)
TILT_FEATURES = {"alpha": "Alpha ratio", "hammarberg": "Hammarberg index"}
# Linear fit; a cubic changes the test RMSE by at most 0.31 dB.
TILT_DEGREE = 1
# time-scale prefix → level name in the tilt baseline's metrics.csv
TILT_LEVELS = {"": "frame", "win_0.26s/": "win_0.26s", "win_0.5s/": "win_0.5s",
               "win_1s/": "win_1s", "leq/": "leq"}


def tilt_summary(fit_dir: Path, feature: str) -> pd.DataFrame:
    """The ``feature`` baseline's metrics at ``TILT_DEGREE``, indexed by (test_set, level)."""
    path = fit_dir / "metrics.csv"
    metrics = pd.read_csv(path)
    if "feature" not in metrics or not (metrics["feature"] == feature).all():
        found = sorted(metrics["feature"].unique()) if "feature" in metrics else "no feature column"
        raise ValueError(f"{path} must hold the {feature} baseline only; found {found}")
    metrics = metrics[metrics["degree"] == TILT_DEGREE].set_index(["test_set", "level"])
    if metrics.index.has_duplicates:
        raise ValueError(f"duplicate (test_set, level) rows in {path}")
    return metrics


def seed_summary(experiments_dir: Path) -> pd.DataFrame:
    """One row per extractor × test set, ``<metric>/mean`` and ``/std`` over the seeds."""
    tidy = to_tidy(collect(experiments_dir))
    summary = aggregate_replicates(tidy, "seed")
    if summary is None:
        raise ValueError(f"no configuration in {experiments_dir} has several seeds")
    if not (summary["n_runs"] == N_SEEDS).all():
        raise ValueError(f"expected {N_SEEDS} runs per configuration, got "
                         f"{sorted(summary['n_runs'].unique())}")
    # to_table keys the rows on every column that is not a value, so drop the unused ones.
    return to_table(summary.drop(columns=["n_runs", "min", "max"]), ["mean", "std"], long=True)


def cell(mean: float, std: float | None, signed: bool) -> str:
    mean = round(mean, 2) + 0.0                                  # + 0.0: no "-0.00"
    sign = "+" if signed and mean >= 0 else ""
    return f"${sign}{mean:.2f}$" if std is None else f"${sign}{mean:.2f} \\pm {std:.2f}$"


def tilt_row(tilt: pd.DataFrame, corpus: str, metric: str, signed: bool) -> list[str]:
    cells = []
    for level in (TILT_LEVELS[prefix] for prefix in SCALES):
        if (corpus, level) not in tilt.index or pd.isna(tilt.loc[(corpus, level), metric]):
            raise ValueError(f"missing tilt baseline {metric} for {corpus} / {level}")
        cells.append(cell(tilt.loc[(corpus, level), metric], None, signed))
    return cells


def latex_table(table: pd.DataFrame, tilts: dict[str, pd.DataFrame]) -> str:
    n_rows = len(TILT_FEATURES) + len(EXTRACTORS)
    rows = []
    for metric, (label, signed) in METRICS.items():
        rows.append(rf"\multicolumn{{{2 + len(SCALES)}}}{{@{{}}l}}{{\emph{{{label} (dB)}}}} \\")
        for corpus, corpus_label in CORPORA.items():
            for i, (feature, feature_label) in enumerate(TILT_FEATURES.items()):
                name = rf"\multirow{{{n_rows}}}{{*}}{{{corpus_label}}}" if i == 0 else ""
                cells = tilt_row(tilts[feature], corpus, metric, signed)
                rows.append(" & ".join([name, feature_label, *cells]) + r" \\")
            for extractor, extractor_label in EXTRACTORS.items():
                match = table[(table["test_set"] == corpus) & (table["extractor.type"] == extractor)]
                if len(match) != 1:
                    raise ValueError(f"{len(match)} rows for {corpus} / {extractor}")
                row = match.iloc[0]
                cells = []
                for prefix in SCALES:
                    mean, std = row[f"{prefix}{metric}/mean"], row[f"{prefix}{metric}/std"]
                    if pd.isna(mean) or pd.isna(std):
                        raise ValueError(f"missing {prefix}{metric} for {corpus} / {extractor}")
                    cells.append(cell(mean, std, signed))
                rows.append(" & ".join(["", extractor_label, *cells]) + r" \\")
            rows.append(r"\addlinespace")
        rows[-1] = r"\midrule" if metric != list(METRICS)[-1] else rows[-1]
    rows.pop()                                                    # no space before \bottomrule
    header = " & ".join(["Corpus", "Feature extractor", *SCALES.values()]) + r" \\"
    return "\n".join([
        "% Generated by intensity_estimation/scripts/make_results_table.py;",
        "% do not edit by hand.",
        rf"\begin{{tabular*}}{{\tblwidth}}{{@{{\extracolsep{{\fill}}}}ll{'c' * len(SCALES)}@{{}}}}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular*}",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments_dir", type=Path, required=True)
    for feature in TILT_FEATURES:
        parser.add_argument(f"--{feature}_baseline", type=Path, required=True,
                            help=f"folder holding the {feature} baseline's metrics.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tilts = {f: tilt_summary(getattr(args, f"{f}_baseline"), f) for f in TILT_FEATURES}
    tex = latex_table(seed_summary(args.experiments_dir), tilts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex)
    print(tex)
    print(f"Written → {args.output}")


if __name__ == "__main__":
    main()
