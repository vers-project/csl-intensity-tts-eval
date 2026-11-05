"""Tests for the attribute table and failure screen, on synthetic run folders."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tts_evaluation import attributes

TEXT = "kids are talking by the door"


def rows(source, experiment, speakers, conditions, sentences=("a", "b"), system="natural"):
    anchor = conditions[0]
    return pd.DataFrame([
        {"utt_id": f"{source}|{s}|{t}|{c}", "group_id": f"{s}|{t}", "condition": c,
         "source": source, "system": system, "experiment": experiment, "speaker_id": s,
         "text": TEXT, "is_anchor": c == anchor}
        for s in speakers for t in sentences for c in conditions
    ])


def write_source(root: Path, job: int, manifest: pd.DataFrame, *, n_frames=None) -> None:
    """One source's manifest, predictor job and speech-eval job, as the real runs lay them out."""
    n_frames = n_frames or {}
    name = manifest["source"].iloc[0]
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    manifest.to_csv(root / "manifests" / f"{name}.csv", index=False)

    ids = manifest["utt_id"]
    pphi_dir = root / "pphi" / str(job) / "predictions"
    pphi_dir.mkdir(parents=True)
    pd.DataFrame({"utt_id": ids, "row_id": range(len(ids)), "source": name,
                  "n_frames": [n_frames.get(u, 50) for u in ids],
                  "leq_pred": 60.0}).to_csv(pphi_dir / "utterances.csv", index=False)

    metrics_dir = root / "metrics" / str(job)
    metrics_dir.mkdir(parents=True)
    pd.DataFrame({"utt_id": ids, "source": name, "text": TEXT, "level.status": "ok",
                  "level.rms_dbfs": -20.0}).to_csv(metrics_dir / "metrics.csv", index=False)

    for run, column in (("pphi", "data.metadata_csv"), ("metrics", "data.manifest")):
        lookup = root / run / "sweep_lookup.csv"
        line = pd.DataFrame({"job_index": [job], column: [f"/x/manifests/{name}.csv"]})
        line.to_csv(lookup, mode="a", header=not lookup.exists(), index=False)


def natural_emotion(root: Path, job: int = 0) -> pd.DataFrame:
    manifest = rows("ravdess", "emotion", ["s0", "s1", "s2", "s3"], ["neutral", "angry"])
    write_source(root, job, manifest)
    return manifest


def loaded(root: Path, source: str) -> pd.DataFrame:
    return attributes.load_source(
        root / "manifests" / f"{source}.csv", attributes.run_jobs(root / "pphi")[source],
        attributes.run_jobs(root / "metrics")[source])


def test_jobs_are_matched_to_sources_by_manifest_name_not_by_order(tmp_path):
    write_source(tmp_path, 3, rows("tts_b", "emotion", ["s0"], ["neutral", "angry"]))
    write_source(tmp_path, 0, rows("tts_a", "emotion", ["s0"], ["neutral", "angry"]))

    jobs = attributes.run_jobs(tmp_path / "metrics")
    assert jobs == {"tts_b": tmp_path / "metrics" / "3", "tts_a": tmp_path / "metrics" / "0"}


def test_the_join_prefixes_pphi_and_keeps_one_copy_of_carried_columns(tmp_path):
    natural_emotion(tmp_path)
    frame = loaded(tmp_path, "ravdess")

    assert len(frame) == 16 and frame["utt_id"].is_unique
    assert {"pphi.n_frames", "pphi.leq_db", "level.rms_dbfs"} <= set(frame.columns)
    assert "row_id" not in frame.columns and "text_x" not in frame.columns


def test_a_table_missing_a_manifest_row_is_refused(tmp_path):
    natural_emotion(tmp_path)
    path = tmp_path / "metrics" / "0" / "metrics.csv"
    pd.read_csv(path).iloc[1:].to_csv(path, index=False)

    with pytest.raises(ValueError, match="1 manifest utt_id missing"):
        loaded(tmp_path, "ravdess")


def test_excluded_rows_are_dropped_and_a_missing_condition_is_refused():
    frame = rows("avid", "effort", ["s0"], ["normal", "loud"])
    frame["excluded"] = [False, True, False, False]
    assert len(attributes.drop_excluded(frame, "avid")) == 3

    frame.loc[0, "condition"] = np.nan
    with pytest.raises(ValueError, match="no condition"):
        attributes.drop_excluded(frame, "avid")


def test_only_a_sample_without_speech_fails(tmp_path):
    natural_emotion(tmp_path)
    frame = loaded(tmp_path, "ravdess")
    frame.loc[frame["utt_id"] == "ravdess|s1|a|angry", "pphi.n_frames"] = 0
    flagged = attributes.flag_failures(frame)
    assert flagged.loc[flagged["failed"], "utt_id"].tolist() == ["ravdess|s1|a|angry"]


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_attribute_table.py"
    spec = importlib.util.spec_from_file_location("build_attribute_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_writes_the_table_and_summaries(tmp_path, monkeypatch):
    natural_emotion(tmp_path, job=0)
    tts = rows("tts_emotion", "emotion", ["s0", "s1"], ["neutral", "angry"], system="tts")
    write_source(tmp_path, 1, tts, n_frames={"tts_emotion|s1|a|angry": 0})
    config = {
        "runs": {"manifests": str(tmp_path / "manifests"), "pphi": str(tmp_path / "pphi"),
                 "metrics": str(tmp_path / "metrics")},
        "sources": ["tts_emotion", "ravdess"],
        "attributes": {"intensity": "pphi.leq_db", "rms": "level.rms_dbfs"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["build_attribute_table.py", "--config",
                                      str(config_path), "--output-dir", str(out)])
    _load_script().main()

    table = pd.read_csv(out / "attributes.csv")
    assert len(table) == 24
    assert set(table.loc[table["failed"], "utt_id"]) == {"tts_emotion|s1|a|angry"}
    summary = pd.read_csv(out / "screen_summary.csv").set_index(["source", "condition"])
    assert summary.loc[("tts_emotion", "angry"), "failed"] == 1
    coverage = pd.read_csv(out / "coverage.csv").set_index("source")
    assert coverage.loc["tts_emotion", "n"] == 8 and coverage["intensity"].sum() == 0
