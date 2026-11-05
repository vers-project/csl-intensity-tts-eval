"""Tests for tts_evaluation.distributions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tts_evaluation import distributions


def table() -> pd.DataFrame:
    rows = []
    for sex, base in (("F", 200.0), ("M", 100.0)):
        for k in range(5):
            rows.append({"utt_id": f"{sex}{k}", "speaker_id": f"{sex}{k % 2}", "sex": sex,
                         "condition": "loud", "x": base + k, "failed": False})
    frame = pd.DataFrame(rows)
    frame.loc[frame["utt_id"] == "F4", "failed"] = True
    frame.loc[frame["utt_id"] == "M4", "x"] = np.nan
    return frame


def test_describe_splits_by_sex_and_skips_failed_rows_and_nans():
    summary = distributions.describe(table(), {"f0": "x"}, ["condition", "sex"],
                                     [0.5]).set_index("sex")
    assert summary.loc["F", "n"] == 4 and summary.loc["M", "n"] == 4
    assert summary.loc["F", "mean"] == pytest.approx(201.5)
    assert summary.loc["M", "q50"] == pytest.approx(101.5)
    assert summary.loc["F", "n_speakers"] == 2


def test_order_by_median_sorts_by_the_natural_median_and_refuses_gaps():
    from tts_evaluation.figures import order_by_median

    summary = pd.DataFrame({
        "condition": ["a", "b", "c", "a", "b", "c"],
        "system": ["natural"] * 3 + ["qwen3tts"] * 3,
        "q50": [70.0, 50.0, 60.0, 10.0, 90.0, 30.0],
    })
    assert order_by_median(summary, ["a", "b", "c"]) == ["b", "c", "a"]
    with pytest.raises(ValueError, match="no natural median"):
        order_by_median(summary, ["a", "b", "d"])


def test_join_utterances_names_the_columns_after_the_run(tmp_path):
    frame = table()
    path = tmp_path / "utterances.csv"
    pd.DataFrame({"utt_id": [*frame["utt_id"], "extra"],
                  "leq_pred": np.arange(len(frame) + 1.0), "alpha": 0.0}).to_csv(path, index=False)
    joined = distributions.join_utterances(frame, path, "tilt", {"leq_pred": "leq_db"})
    assert joined["tilt.leq_db"].tolist() == list(np.arange(len(frame), dtype=float))
    assert "alpha" not in joined.columns and len(joined) == len(frame)
    pd.DataFrame({"utt_id": frame["utt_id"][1:], "leq_pred": 0.0}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="1 utt_id"):
        distributions.join_utterances(frame, path, "tilt", {"leq_pred": "leq_db"})
