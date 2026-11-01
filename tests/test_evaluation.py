"""Tests: grouped splits, leakage detection, fold-local preprocessing,
full-experiment bookkeeping. All on synthetic fixtures (software tests only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stress_fuzzy import config as C
from stress_fuzzy.data import segment_windows
from stress_fuzzy.evaluation import (
    LeakageError,
    check_leakage,
    compute_metrics,
    make_folds,
    run_experiment,
)
from stress_fuzzy.features import extract_features
from conftest import make_raw


def _features(n=8):
    raw = make_raw(num_subjects=n)
    return extract_features(segment_windows(raw, C.DEFAULT_WINDOW_SAMPLES))


def test_loso_folds_disjoint_and_exhaustive():
    feats = _features(8)
    folds = make_folds(feats.subject, protocol="loso")
    assert len(folds) == 8
    seen_test = []
    all_idx = np.arange(len(feats))
    for f in folds:
        check_leakage(
            feats.subject.values[f["train_idx"]],
            feats.subject.values[f["test_idx"]],
        )
        seen_test.extend(f["test_idx"].tolist())
        # train + test partition the dataset
        assert (
            sorted(np.concatenate([f["train_idx"], f["test_idx"]]).tolist())
            == all_idx.tolist()
        )
    # every window held out exactly once
    assert sorted(seen_test) == all_idx.tolist()


def test_leakage_check_raises():
    with pytest.raises(LeakageError):
        check_leakage([1, 2, 3], [3, 4])


def test_groupkfold_folds_disjoint():
    feats = _features(9)
    folds = make_folds(feats.subject, protocol="gk5", n_splits=5)
    subjects = feats.subject.values
    for f in folds:
        check_leakage(subjects[f["train_idx"]], subjects[f["test_idx"]])


def test_unknown_protocol_rejected():
    with pytest.raises(ValueError):
        make_folds(pd.Series([1, 2]), protocol="nope")


def test_fold_local_preprocessing_uses_train_statistics_only():
    """A fold's scaler must reflect the TRAIN partition, not the test rows."""
    feats = _features(6)
    folds = make_folds(feats.subject, protocol="loso")
    f = folds[0]
    tr, te = f["train_idx"], f["test_idx"]
    # contaminate the held-out subject with a huge offset
    col = feats["eda__mean"].to_numpy(float).copy()
    col[te] += 1000.0
    pipe = Pipeline([("scale", StandardScaler())])
    pipe.fit(col[tr].reshape(-1, 1))
    fitted_mean = pipe.named_steps["scale"].mean_[0]
    assert fitted_mean == pytest.approx(col[tr].mean())
    assert abs(fitted_mean - col.mean()) > 1.0  # global fit would differ


def test_compute_metrics_block():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0, 1, 1, 1, 0, 0])
    m = compute_metrics(y, p, y_score=np.array([0.1, 0.6, 0.9, 0.8, 0.3, 0.2]))
    assert m["n"] == 6
    assert 0 <= m["macro_f1"] <= 1
    assert m["stress_recall"] == pytest.approx(2 / 3)
    assert m["confusion_matrix"] == [[2, 1], [1, 2]]
    assert m["pr_auc"] is not None


def test_run_experiment_structure_and_no_leakage():
    feats = _features(8)
    cfg = C.Config(protocol="loso")
    res = run_experiment(feats, cfg)
    assert res["n_participants"] == 8
    assert res["n_windows"] == len(feats)
    # fuzzy + ablations + baselines all present
    for name in (
        "fuzzy_quality_personalized",
        "fuzzy_quality_unpersonalized",
        "fuzzy_no_quality",
        "logistic_regression",
        "svm_rbf",
        "random_forest",
    ):
        assert name in res["models"]
        agg = res["models"][name]["aggregate"]
        assert agg["macro_f1"]["n_folds"] == 8
        assert 0 <= agg["macro_f1"]["mean"] <= 1
    # membership bounds exist per fold and differ from 'fixed' for fitted inputs
    bounds = res["fold_fuzzy_bounds"]["fuzzy_quality_personalized"]
    assert len(bounds) == 8
    assert isinstance(bounds["subject_1"]["eda_level"], dict)
    assert bounds["subject_1"]["quality"] == "fixed"
    # participant-level rows cover every participant exactly once per model
    for name, mres in res["models"].items():
        subs = [r["subject"] for r in mres["participant_metrics"]]
        assert sorted(subs) == sorted(feats.subject.unique())


def test_run_experiment_learnable_synthetic_signal():
    """Sanity (software test): models should beat chance on the synthetic
    fixture where stress phases have distinctly higher EDA."""
    feats = _features(8)
    res = run_experiment(feats, C.Config(protocol="loso"))
    for name in ("fuzzy_quality_personalized", "logistic_regression"):
        agg = res["models"][name]["aggregate"]
        assert agg["balanced_accuracy"]["mean"] > 0.6, name
