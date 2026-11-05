"""Collect test metrics and configs from all experiments into a single CSV.

Usage
-----
    python collect_experiment_results.py \
        --experiments_dir /path/to/grid \
        --output results.csv \
        [--long] [--metrics me_db rmse_db r2] [--replicate-key seed] [--per-run] [--raw]

For each experiment directory (leaf dirs containing config.yaml + version_*/metrics.csv),
the script extracts the last row containing test metrics and flattens the YAML config
into columns. Nested config keys are joined with '.' (e.g. extractor.type, model.n_layers).

Nothing in the script knows about a particular sweep.  By default the table is made
readable:

* only the config keys whose value differs between the collected runs are kept, so each
  row is identified by what distinguishes it (works across several grids too);
* Lightning's ``/dataloader_idx_N`` suffix is stripped from the metric names;
* ``experiment`` is the run directory relative to ``--experiments_dir``.

**Replicates are aggregated.**  When ``--replicate-key`` (default ``seed``) is one of the
keys that differ and some configuration has more than one run, the runs that share every
other key become one row with ``n_runs``, ``experiments`` (the run directories) and, for
every metric, ``<metric>/min``, ``/max``, ``/mean`` and ``/std`` (sample std, ddof = 1).
``--per-run`` keeps one row per run instead.

``--long`` gives one row per run (or configuration) × test set, splitting each metric name
``test/<set>/<rest>`` into a ``test_set`` row and a ``<rest>`` column.  ``--metrics`` keeps
only the metrics whose last ``/`` component is named (e.g. ``me_db rmse_db``).

``--source predictions`` reads each run's ``predictions/metrics_dataset.csv`` (written by
``scripts/analyze_predictions.py``) instead of Lightning's ``metrics.csv``; its metrics are
named ``test/<test_set>/<level>/<metric>``, the level being ``frame``, ``leq`` or a
sliding window ``win_<w>s``.  Runs without that file are skipped.

``--raw`` restores the original output -- every config key, suffixed metric names, absolute
``experiment_dir``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

_DATALOADER_SUFFIX = re.compile(r"/dataloader_idx_\d+$")


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict with dot-separated keys."""
    out = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, full_key))
        else:
            out[full_key] = v
    return out


def find_metrics_csv(exp_dir: Path) -> Path | None:
    """Return the metrics.csv inside version_*/ subdirectory, or None."""
    for version_dir in sorted(exp_dir.glob("version_*")):
        candidate = version_dir / "metrics.csv"
        if candidate.exists():
            return candidate
    return None


def extract_test_row(metrics_csv: Path, strip_dataloader: bool = True) -> dict | None:
    """Return the last row that has at least one non-NaN test/* column."""
    df = pd.read_csv(metrics_csv)
    test_cols = [c for c in df.columns if c.startswith("test/")]
    if not test_cols:
        return None
    test_df = df.dropna(subset=test_cols, how="all")
    if test_df.empty:
        return None
    row = test_df.iloc[-1]
    rename = (lambda c: _DATALOADER_SUFFIX.sub("", c)) if strip_dataloader else (lambda c: c)
    return {rename(c): row[c] for c in test_cols if pd.notna(row[c])}


def extract_prediction_metrics(exp_dir: Path) -> dict | None:
    """``{"test/<test_set>/<level>/<metric>": value}`` from ``predictions/metrics_dataset.csv``."""
    path = exp_dir / "predictions" / "metrics_dataset.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    metrics = [c for c in table.columns if c not in ("test_set", "level")]
    return {
        f"test/{row['test_set']}/{row['level']}/{metric}": row[metric]
        for _, row in table.iterrows() for metric in metrics if pd.notna(row[metric])
    }


def collect(experiments_dir: Path, raw: bool = False, source: str = "lightning") -> pd.DataFrame:
    records = []

    config_files = sorted(experiments_dir.rglob("config.yaml"), key=_natural_key)
    for config_path in config_files:
        exp_dir = config_path.parent

        if source == "predictions":
            test_metrics = extract_prediction_metrics(exp_dir)
        else:
            metrics_csv = find_metrics_csv(exp_dir)
            if metrics_csv is None:
                continue
            test_metrics = extract_test_row(metrics_csv, strip_dataloader=not raw)
        if not test_metrics:
            continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

        flat_cfg = flatten_dict(cfg)

        # Remove launcher/slurm keys — not useful for analysis
        flat_cfg = {k: v for k, v in flat_cfg.items() if not k.startswith("launcher")}

        exp_name = str(exp_dir) if raw else str(exp_dir.relative_to(experiments_dir))
        record = {
            "experiment_dir" if raw else "experiment": exp_name,
            **test_metrics,
            **flat_cfg,
        }
        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    id_col = df.columns[0]
    test_cols = [c for c in df.columns if c.startswith("test/")]
    config_cols = [c for c in df.columns if c not in [id_col] + test_cols]
    if raw:
        return df[[id_col] + test_cols + config_cols]
    # The run id, then what distinguishes the runs, then the metrics
    return df[[id_col] + varying_columns(df, config_cols) + test_cols]


def varying_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """The columns whose value is not the same in every row.

    Compared through ``repr`` so list-valued keys and keys missing from some runs (NaN)
    count as values like any other.
    """
    return [c for c in columns if df[c].map(repr).nunique() > 1]


def to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run × metric: the id and config columns, ``metric`` and ``value``.

    List-valued keys (e.g. conv dilations) become strings so they can be grouped on.
    """
    metric_cols = [c for c in df.columns if c.startswith("test/")]
    id_cols = [c for c in df.columns if c not in metric_cols]
    df = df.assign(**{
        c: df[c].map(lambda v: str(v) if isinstance(v, list) else v) for c in id_cols
    })
    return df.melt(id_vars=id_cols, value_vars=metric_cols, var_name="metric").dropna(
        subset=["value"]
    )


STATS = ("min", "max", "mean", "std")


def aggregate_replicates(tidy: pd.DataFrame, key: str) -> pd.DataFrame | None:
    """Group the runs that differ only by ``key`` and summarise every metric over them.

    The configuration is every column but ``experiment``, ``key``, ``metric`` and
    ``value``.  Returns one row per configuration × metric with ``n_runs``,
    ``experiments`` (the run directories, space-separated) and ``min`` / ``max`` /
    ``mean`` / ``std`` (sample std, ddof = 1) -- or ``None`` when ``key`` does not vary
    or no configuration has more than one run.
    """
    if key not in tidy.columns:
        return None
    config = [c for c in tidy.columns if c not in ("experiment", key, "metric", "value")]
    groups = tidy.groupby(config + ["metric"], dropna=False, sort=False)
    runs = groups["experiment"].agg(
        n_runs="size", experiments=lambda e: " ".join(map(str, e))
    )
    if runs["n_runs"].max() < 2:
        return None
    return runs.join(groups["value"].agg(list(STATS))).reset_index()


def to_table(tidy: pd.DataFrame, values: list[str], long: bool) -> pd.DataFrame:
    """Pivot the tidy rows back to one column per metric (× statistic).

    ``long`` splits ``test/<set>/<rest>`` into a ``test_set`` row key and a ``<rest>``
    column, so each test set gets its own row.  With several ``values`` (the statistics)
    a column is ``<metric>/<statistic>``.
    """
    tidy = tidy.copy()
    if long:
        parts = tidy["metric"].str.split("/", n=2)
        tidy["test_set"], tidy["metric"] = parts.str[1], parts.str[2]
    index = [c for c in tidy.columns if c not in ["metric", *values]]
    # set_index + unstack, not pivot_table: it keeps NaN config values (a key absent from
    # some runs) and never builds the cross product of the index levels.
    table = tidy.set_index(index + ["metric"])[values].unstack("metric")
    metrics = list(dict.fromkeys(tidy["metric"]))
    if len(values) == 1:
        table.columns = table.columns.droplevel(0)
        table = table[metrics]
    else:
        table = table[[(v, m) for m in metrics for v in values]]
        table.columns = [f"{m}/{v}" for v, m in table.columns]
    return table.reset_index()


def _natural_key(path: Path) -> list:
    """Sort run directories 0, 1, 2, …, 10 rather than 0, 1, 10, 11, …, 2."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(path))]


def main():
    parser = argparse.ArgumentParser(description="Collect experiment results into a CSV.")
    parser.add_argument(
        "--experiments_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_results.csv"),
    )
    parser.add_argument("--long", action="store_true",
                        help="one row per run × test set instead of one row per run")
    parser.add_argument("--replicate-key", default="seed",
                        help="config key whose runs are aggregated (default: seed)")
    parser.add_argument("--per-run", action="store_true",
                        help="never aggregate: one row per run")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="keep only these metrics, e.g. me_db rmse_db r2")
    parser.add_argument("--source", choices=("lightning", "predictions"), default="lightning",
                        help="Lightning's metrics.csv, or predictions/metrics_dataset.csv")
    parser.add_argument("--raw", action="store_true",
                        help="original output: all config keys, suffixed metric names "
                             "(what make_latex_table.py reads)")
    args = parser.parse_args()
    if args.raw and (args.long or args.metrics or args.per_run):
        parser.error("--raw is the original wide output; it cannot be combined with "
                     "--long, --metrics or --per-run.")

    print(f"Scanning: {args.experiments_dir}")
    df = collect(args.experiments_dir, raw=args.raw, source=args.source)

    if df.empty:
        print("No completed experiments found.")
        return

    if not args.raw:
        tidy = to_tidy(df)
        if args.metrics:
            tidy = tidy[tidy["metric"].str.split("/").str[-1].isin(args.metrics)]
        summary = None if args.per_run else aggregate_replicates(tidy, args.replicate_key)
        if summary is None:
            df = to_table(tidy, ["value"], args.long)
        else:
            print(f"Aggregated over '{args.replicate_key}': "
                  f"{summary['n_runs'].min()}-{summary['n_runs'].max()} runs per configuration.")
            df = to_table(summary, list(STATS), args.long)

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows → {args.output}")
    if not args.raw:
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
