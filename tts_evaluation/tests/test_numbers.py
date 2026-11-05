"""Tests for tts_evaluation.numbers."""
from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from tts_evaluation import numbers

EFFORT = ["soft", "normal", "loud", "veryloud"]
EMOTION = ["neutral", "angry"]


def summary() -> pd.DataFrame:
    medians = {"natural": [51.25, 58.1, 64.9, 68.84], "cosyvoice3": [53.2, 58.0, 62.3, 66.3],
               "indextts2": [52.5, 55.5, 59.0, 63.0], "qwen3tts": [54.9, 61.6, 72.8, 72.5]}
    emotion = {"natural": [57.0, 73.9], "cosyvoice3": [58.4, 73.8],
               "indextts2": [59.1, 72.3], "qwen3tts": [61.8, 73.3]}
    rows = []
    for experiment, conds, table in (("effort", EFFORT, medians), ("emotion", EMOTION, emotion)):
        for system, values in table.items():
            for condition, q50 in zip(conds, values):
                rows.append({"experiment": experiment, "system": system, "condition": condition,
                             "sex": "all", "attribute": "intensity", "q50": q50,
                             "q05": q50 - 5, "q95": q50 + 5})
                rows.append({"experiment": experiment, "system": system, "condition": condition,
                             "sex": "F", "attribute": "intensity", "q50": 0.0,
                             "q05": 0.0, "q95": 0.0})
    return pd.DataFrame(rows)


def compute():
    return numbers.paper_numbers(summary(), {"effort": EFFORT, "emotion": EMOTION},
                                 {"QwenLoudNaturalVeryLoud": (["effort", "qwen3tts", "loud"],
                                                              ["effort", "natural", "veryloud"])})


def test_medians_round_half_up_and_differences_use_the_rounded_medians():
    out = compute()
    assert out["medEffNaturalSoft"] == Decimal("51.3")          # 51.25 → 51.3, not 51.2
    assert out["medEffNaturalVeryLoud"] == Decimal("68.8")
    # 72.8 − 68.8 on the page, whatever the unrounded medians give
    assert out["diffQwenLoudNaturalVeryLoud"] == Decimal("4.0")
    assert out["gapEffCosySoft"] == Decimal("1.9")              # 53.2 − 51.3
    assert out["stepQwenVeryLoud"] == Decimal("0.3")            # |72.5 − 72.8|
    assert out["rangeEffNatural"] == Decimal("17.5")
    assert out["rangeEmoIndex"] == Decimal("13.2")
    assert out["lowEffNatural"] == Decimal("46.3") and out["highEffQwen"] == Decimal("77.8")


def test_only_pooled_rows_are_read_and_names_are_letters_only():
    out = compute()
    assert all(name.isalpha() for name in out)
    assert "gapEffNaturalSoft" not in out


def test_a_missing_cell_stops_the_script():
    data = summary()
    data = data[~((data["system"] == "qwen3tts") & (data["condition"] == "loud"))]
    with pytest.raises(ValueError, match="no median"):
        numbers.paper_numbers(data, {"effort": EFFORT, "emotion": EMOTION}, {})


def test_maxgap_is_the_largest_gap_of_the_system():
    out = compute()
    gaps = [out[f"gapEffCosy{c}"] for c in ("Soft", "Normal", "Loud", "VeryLoud")]
    assert out["maxgapEffCosy"] == max(gaps) == Decimal("2.6")      # loud: 64.9 − 62.3
    assert "maxgapEffNatural" not in out


def test_failures_follow_the_source_queries_and_exclusions():
    rows = []
    for k in range(4):
        rows.append({"source": "ravdess", "experiment": "emotion", "system": "natural",
                     "condition": "angry", "intensity": "strong" if k < 2 else "normal",
                     "failed": k == 0})
    for k in range(10):
        rows.append({"source": "indextts2_emotion", "experiment": "emotion",
                     "system": "indextts2", "condition": "angry", "intensity": None,
                     "failed": k < 3})
        rows.append({"source": "indextts2_emotion", "experiment": "emotion",
                     "system": "indextts2", "condition": "calm", "intensity": None,
                     "failed": k < 1})
    counts = numbers.failure_counts(pd.DataFrame(rows), {
        "ravdess": "condition == 'neutral' or intensity == 'strong'", "indextts2_emotion": None})
    out = numbers.failure_numbers(counts, [["emotion", "indextts2", "angry"]])
    assert (out["failNEmoNaturalAngry"], out["failTotalEmoNaturalAngry"]) == (1, 2)
    assert out["failRateEmoNaturalAngry"] == Decimal("50.0")
    assert out["failRateEmoIndexAngry"] == Decimal("30.0")
    assert out["failMaxRateOther"] == Decimal("10.0")            # angry is excluded
    assert out["failMaxRateNatural"] == Decimal("50.0")


def test_alpha_agreement_reads_order_and_compression():
    data = summary()
    pred = data[data["attribute"] == "intensity"]
    # Alpha = predictor halved around 60 dB (same order), except Qwen3-TTS effort, whose
    # alpha medians reverse soft and normal.
    alpha = pred.assign(attribute="alpha", q50=lambda d: 60 + (d["q50"] - 60) / 2)
    swap = (alpha["system"] == "qwen3tts") & (alpha["experiment"] == "effort")
    alpha.loc[swap & (alpha["condition"] == "soft"), "q50"] = 59.0
    alpha.loc[swap & (alpha["condition"] == "normal"), "q50"] = 57.0
    out = numbers.alpha_agreement(pd.concat([data, alpha]), {"effort": EFFORT,
                                                             "emotion": EMOTION}, "alpha")
    assert out["spearmanEffNatural"] == Decimal("1.00")
    assert out["spearmanEffQwen"] == Decimal("0.80")            # one adjacent swap of 4
    assert out["rangeEffNaturalAlpha"] == Decimal("8.8")         # 64.4 − 55.6, rounded medians
    assert out["alphaShareEffMin"] == Decimal("50")              # 8.8 / 17.5
    assert out["alphaShareEmoMax"] == Decimal("50")


def test_calibrated_numbers_correlate_samples_with_the_label():
    rows = pd.DataFrame({"condition": ["soft", "soft", "loud", "loud", "loud"],
                         "label": [50.0, 52.0, 70.0, 68.0, None],
                         "pred": [51.0, 53.0, 71.0, 69.0, 60.0],
                         "alpha": [55.0, 50.0, 60.0, 65.0, 60.0]})
    out = numbers.calibrated_numbers(rows, ["soft", "loud"], "label", "pred", "alpha")
    assert out["corrLabelPredictor"] == Decimal("1.00")          # pred = label + 1
    assert out["rangeEffNaturalLabel"] == Decimal("18.0")        # 69.0 − 51.0, NaN row dropped
    assert Decimal("0") < out["corrLabelAlpha"] < Decimal("1")
    with pytest.raises(ValueError, match="conditions"):
        numbers.calibrated_numbers(rows, ["soft", "normal", "loud"], "label", "pred", "alpha")


def test_rms_spread_is_the_range_of_the_rounded_medians():
    data = summary()
    rms = data[data["attribute"] == "intensity"].assign(attribute="rms",
                                                        q50=lambda d: d["q50"] - 80.0)
    out = numbers.rms_numbers(pd.concat([data, rms]), {"effort": EFFORT, "emotion": EMOTION},
                              "rms")
    assert out["rmsSpreadEffNatural"] == Decimal("17.6")          # −11.2 − (−28.8)
    assert out["rmsSpreadEffQwen"] == Decimal("17.9")             # loud is the maximum
