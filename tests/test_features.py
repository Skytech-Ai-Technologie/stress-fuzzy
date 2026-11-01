"""Tests: feature statistics and quality flags on hand-computable windows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stress_fuzzy import config as C
from stress_fuzzy.data import segment_windows
from stress_fuzzy.features import channel_stats, extract_features, window_quality
from conftest import make_raw


def test_channel_stats_known_values():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    s = channel_stats(v)
    assert s["mean"] == 2.5
    assert s["min"] == 1.0 and s["max"] == 4.0
    assert s["median"] == 2.5
    assert abs(s["iqr"] - 1.5) < 1e-9  # linear quartiles 1.75 / 3.25
    assert abs(s["slope"] - 1.0) < 1e-9  # perfect ramp


def test_channel_stats_all_nan():
    s = channel_stats(np.array([np.nan, np.nan]))
    assert all(np.isnan(s[k]) for k in s)


def test_window_quality_clean():
    df = pd.DataFrame(
        {
            "eda": np.full(56, 0.5),
            "bvp": np.linspace(-20, 20, 56),
            "temp": np.full(56, 33.0),
            "x": np.zeros(56),
            "y": np.zeros(56),
            "z": np.zeros(56),
        }
    )
    q = window_quality(df)
    assert q["flat_channels"] == ["eda", "temp", "x", "y", "z"]
    assert 0.0 <= q["quality"] < 1.0  # flatline penalised but plausible


def test_window_quality_implausible_and_missing():
    df = pd.DataFrame(
        {
            "eda": np.append(np.full(55, 0.5), 500.0),   # >100 out of range
            "bvp": np.full(56, 0.0),
            "temp": np.full(56, 33.0),
            "x": np.full(56, 1.0),
            "y": np.full(56, 1.0),
            "z": np.full(56, 9.0),
        }
    )
    df.loc[3, "temp"] = np.nan
    q = window_quality(df)
    assert q["implausible_fraction"] > 0
    assert q["missing_fraction"] > 0
    assert q["quality"] < 1.0


def test_extract_features_shape_and_ranges():
    raw = make_raw(num_subjects=3)
    widx = segment_windows(raw, C.DEFAULT_WINDOW_SAMPLES)
    feats = extract_features(widx)
    assert len(feats) == len(widx)
    assert set(["subject", "label", "eda__mean", "bvp__std", "quality"]).issubset(
        feats.columns
    )
    assert feats.quality.between(0, 1).all()
    # synthetic stress windows must have higher mean EDA than non-stress
    med = feats.groupby("label")["eda__mean"].mean()
    assert med[1] > med[0]
