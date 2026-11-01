"""Shared synthetic fixtures.

IMPORTANT: these fixtures exist ONLY to test the software. They are not the
WorkStress3D dataset, they contain no real measurements, and any numbers
produced from them must never be presented as scientific results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stress_fuzzy import config as C

WINDOW = C.DEFAULT_WINDOW_SAMPLES  # 56


def make_raw(num_subjects: int = 8, subject_col: str = "Subject",
             seed: int = 0) -> pd.DataFrame:
    """Synthetic physiological stream: alternating 0/1 phases whose length is
    a multiple of WINDOW; stress phases have higher EDA/BVP variability."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(1, num_subjects + 1):
        base_eda = 0.3 + 0.1 * s
        blocks = []
        for phase, label in enumerate([0, 1, 0, 1]):
            n = WINDOW * 2  # equal windows per label so per-subject stats are balanced
            eda = rng.normal(base_eda + (1.0 if label else 0.0), 0.05, n)
            bvp = rng.normal(0, 25 if label else 10, n)
            temp = rng.normal(33.0 + (0.4 if label else 0.0), 0.1, n)
            x = rng.normal(10, 2, n)
            y = rng.normal(0, 2, n)
            z = rng.normal(20, 2, n)
            blocks.append(
                pd.DataFrame(
                    {
                        subject_col: s,
                        "eda": eda,
                        "bvp": bvp,
                        "temp": temp,
                        "x": x,
                        "y": y,
                        "z": z,
                        "emotion": label,
                    }
                )
            )
        frames.append(pd.concat(blocks, ignore_index=True))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def synthetic_data_dir(tmp_path) -> object:
    """Minimal WorkStress3D-shaped directory from synthetic data."""
    for name, alias in [("15sn", "Subject"), ("30sn", "subjet"), ("60sn", "Subject")]:
        raw = make_raw(num_subjects=6, subject_col=alias, seed=1)
        d = tmp_path / "PhysiologicalSignals"
        d.mkdir(exist_ok=True)
        raw.to_csv(d / f"physiological_signals_{name}.csv", index=False)
    sv = tmp_path / "Survey"
    sv.mkdir()
    pd.DataFrame({"Subject": range(1, 7)}).to_csv(sv / "demographic.csv", index=False)
    pd.DataFrame(
        {"Subject": [1, 1], "C1": ["Tension", "Good"]}
    ).to_csv(sv / "instantQuestionnaires.csv", index=False)
    pd.DataFrame({"Subject": range(1, 7), "TotalPoints": range(30, 36)}).to_csv(
        sv / "generalStressTest.csv", index=False
    )
    aud = tmp_path / "AudioData"
    aud.mkdir()
    (aud / "ses1.m4a").write_bytes(b"x")
    fx = tmp_path / "TheFacialExpressions"
    fx.mkdir()
    pd.DataFrame({"emotion": [0, 1], "pixels": ["1 2 3", "4 5 6"]}).to_csv(
        fx / "facial_expression.csv", index=False
    )
    return tmp_path
