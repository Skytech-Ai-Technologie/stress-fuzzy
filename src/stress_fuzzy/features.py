"""Window feature extraction: interpretable statistics + signal-quality flags.

Per window and channel (eda, bvp, temp, x, y, z):
    mean, std, median, iqr, min, max, slope (OLS vs sample index)

Domain/quality features:
    eda_dynamic_range, acc_magnitude_mean, acc_magnitude_std (motion proxy),
    temp_deviation_from_reference, and per-window quality flags:
    missing_fraction, flatline channels, implausible-sample fraction,
    motion score, overall quality in [0, 1] (1 = clean).

Every feature name is ``<channel>__<stat>`` so the provenance of each model
input is readable in reports and explanations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import WindowIndex

FLATLINE_STD_EPS = 1e-6  # a channel with std below this over a window is flat
TEMP_REFERENCE = 33.0     # degrees C; wrist-skin population reference (design choice)


def _slope(values: np.ndarray) -> float:
    """Ordinary-least-squares slope of the values against sample index."""
    n = len(values)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=float)
    denom = np.sum((t - t.mean()) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((t - t.mean()) * (values - values.mean())) / denom)


def channel_stats(values: np.ndarray) -> dict:
    """Robust, interpretable statistics for one channel window."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return {k: np.nan for k in
                ("mean", "std", "median", "iqr", "min", "max", "slope")}
    q25, q75 = np.percentile(finite, [25, 75])
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "iqr": float(q75 - q25),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "slope": _slope(finite),
    }


def window_quality(sample: pd.DataFrame) -> dict:
    """Quality flags for one window; quality in [0,1] (1 = clean signal).

    Penalties (subtracted from 1.0):
      - 0.35 * missing fraction
      - 0.25 * fraction of samples outside plausible sensor ranges
      - 0.25 if any channel is flatlined
      - 0.15 * min(1, motion/60) where motion is the std of the ACC magnitude
    """
    n = len(sample)
    missing_frac = float(sample.isna().to_numpy().mean()) if n else 1.0

    implausible = 0
    total = 0
    for col in C.SIGNAL_COLUMNS:
        lo, hi = C.PHYSIO_RANGES[col]
        v = pd.to_numeric(sample[col], errors="coerce").to_numpy(float)
        ok = np.isfinite(v)
        total += int(ok.sum())
        implausible += int((((v < lo) | (v > hi)) & ok).sum())
    implausible_frac = implausible / total if total else 1.0

    flat_channels = []
    for col in C.SIGNAL_COLUMNS:
        v = pd.to_numeric(sample[col], errors="coerce").to_numpy(float)
        if np.isfinite(v).all() and v.size > 1 and float(np.std(v)) < FLATLINE_STD_EPS:
            flat_channels.append(col)

    acc_mag = np.linalg.norm(
        sample[C.ACC_COLUMNS].to_numpy(dtype=float), axis=1
    )
    motion = float(np.std(acc_mag)) if acc_mag.size > 1 else 0.0

    quality = 1.0
    quality -= 0.35 * missing_frac
    quality -= 0.25 * implausible_frac
    quality -= 0.25 if flat_channels else 0.0
    quality -= 0.15 * min(1.0, motion / 60.0)
    quality = float(np.clip(quality, 0.0, 1.0))

    return {
        "missing_fraction": missing_frac,
        "implausible_fraction": float(implausible_frac),
        "flat_channels": flat_channels,
        "motion": motion,
        "quality": quality,
    }


def extract_features(widx: WindowIndex) -> pd.DataFrame:
    """One row per window: metadata + channel stats + quality features."""
    rows = []
    for i in range(len(widx.table)):
        meta = widx.table.iloc[i]
        sample = widx.window_samples_frame(i)
        row = {
            "subject": int(meta.subject),
            "phase_id": int(meta.phase_id),
            "window": int(meta.window),
            "label": int(meta.label),
        }
        for col in C.SIGNAL_COLUMNS:
            stats = channel_stats(sample[col].to_numpy(dtype=float))
            for stat, val in stats.items():
                row[f"{col}__{stat}"] = val
        row["eda_dynamic_range"] = row["eda__max"] - row["eda__min"]
        acc_mag = np.linalg.norm(sample[C.ACC_COLUMNS].to_numpy(dtype=float), axis=1)
        row["acc_magnitude_mean"] = float(np.mean(acc_mag))
        row["acc_magnitude_std"] = float(np.std(acc_mag))
        row["temp_deviation"] = abs(row["temp__mean"] - TEMP_REFERENCE)
        q = window_quality(sample)
        row.update(
            {
                "quality__missing_fraction": q["missing_fraction"],
                "quality__implausible_fraction": q["implausible_fraction"],
                "quality__flat_channels": ",".join(q["flat_channels"]),
                "quality__motion": q["motion"],
                "quality": q["quality"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    f"{c}__{s}"
    for c in C.SIGNAL_COLUMNS
    for s in ("mean", "std", "median", "iqr", "min", "max", "slope")
] + [
    "eda_dynamic_range",
    "acc_magnitude_mean",
    "acc_magnitude_std",
    "temp_deviation",
    "quality__missing_fraction",
    "quality__implausible_fraction",
    "quality__motion",
    "quality",
]
