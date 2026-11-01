"""Tests: membership functions, rule evaluation, quality gating, abstention,
missing-modality behaviour, and train-only threshold fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stress_fuzzy import config as C
from stress_fuzzy.data import segment_windows
from stress_fuzzy.features import extract_features
from stress_fuzzy.fuzzy_model import (
    FuzzyStressClassifier,
    FuzzyEngine,
    FuzzyOutput,
    apply_mf,
    build_fuzzy_inputs,
    load_rules,
    parse_rules,
    partition_3,
    trapezoid,
    triangle,
)
from conftest import make_raw


# ------------------------------------------------------------- membership


def test_triangle_endpoints_and_peak():
    assert triangle(0.5, 0.0, 0.5, 1.0) == 1.0
    assert triangle(0.0, 0.0, 0.5, 1.0) == 0.0
    assert triangle(1.0, 0.0, 0.5, 1.0) == 0.0
    assert triangle(0.25, 0.0, 0.5, 1.0) == pytest.approx(0.5)
    assert triangle(2.0, 0.0, 0.5, 1.0) == 0.0  # outside


def test_trapezoid_plateau_and_shoulders():
    spec = {"type": "trapezoid", "params": [0.5, 0.75, 1.0, 1.0]}
    assert apply_mf(0.9, spec) == 1.0
    assert apply_mf(1.5, spec) == 1.0   # saturating shoulder
    assert apply_mf(0.5, spec) == 0.0
    assert apply_mf(0.625, spec) == pytest.approx(0.5)


def test_partition_3_covers_domain():
    part = partition_3(None, 0.0, 1.0, 2.0, 3.0, 4.0)
    xs = np.linspace(-0.5, 4.5, 201)
    total = np.zeros_like(xs)
    for t in ("low", "medium", "high"):
        total += apply_mf(xs, part[t])
    # every point inside the domain belongs to at least one term
    inside = (xs >= 0) & (xs <= 4)
    assert (total[inside] > 0).all()
    assert (total <= 1.0 + 1e-9).all()


# ----------------------------------------------------------------- engine


def _tiny_engine() -> FuzzyEngine:
    spec = load_rules()
    input_terms = {
        "eda_level": {
            "low": {"type": "trapezoid", "params": [-3, -3, -1, 0]},
            "medium": {"type": "triangle", "params": [-1, 0, 1]},
            "high": {"type": "trapezoid", "params": [0, 1, 3, 3]},
        },
        "bvp_variability": {
            "low": {"type": "trapezoid", "params": [-3, -3, -1, 0]},
            "medium": {"type": "triangle", "params": [-1, 0, 1]},
            "high": {"type": "trapezoid", "params": [0, 1, 3, 3]},
        },
        "temp_deviation": {
            "low": {"type": "trapezoid", "params": [-3, -3, -1, 0]},
            "medium": {"type": "triangle", "params": [-1, 0, 1]},
            "high": {"type": "trapezoid", "params": [0, 1, 3, 3]},
        },
        "motion": {
            "low": {"type": "trapezoid", "params": [-3, -3, -1, 0]},
            "medium": {"type": "triangle", "params": [-1, 0, 1]},
            "high": {"type": "trapezoid", "params": [0, 1, 3, 3]},
        },
        "quality": dict(spec["fixed_inputs"]["quality"]["terms"]),
    }
    rules = parse_rules(spec["rules"])
    return FuzzyEngine(input_terms, spec["output_terms"], rules)


def test_engine_high_stress_inputs():
    eng = _tiny_engine()
    out = eng.infer(
        {
            "eda_level": 2.0,
            "bvp_variability": 2.0,
            "temp_deviation": 0.5,
            "motion": -2.0,
            "quality": 0.95,
        }
    )
    assert out.score is not None and out.score > 0.6
    assert any(r["id"] == "stress_eda_bvp" for r in out.fired_rules)
    assert out.stress_mass > 0


def test_engine_low_quality_fires_uncertain_not_stress():
    eng = _tiny_engine()
    out = eng.infer(
        {
            "eda_level": 2.0,
            "bvp_variability": 2.0,
            "temp_deviation": 2.0,
            "motion": 2.0,
            "quality": 0.1,  # poor
        }
    )
    assert out.uncertain_mass > 0
    assert any(r["consequent"] == "uncertain" for r in out.fired_rules)
    # the quality antecedent 'good' is unmet, so no high-stress rule fires
    assert all(r["consequent"] != "high" for r in out.fired_rules)


def test_engine_missing_input_skips_only_dependent_rules():
    eng = _tiny_engine()
    out = eng.infer(
        {"eda_level": -2.0, "bvp_variability": -2.0, "quality": 0.9}
    )  # temp_deviation and motion absent -> "missing modality"
    assert out.skipped_rules, "rules referencing absent inputs must be skipped"
    # calm_eda_bvp needs only present inputs and matches them -> must fire
    assert any(r["id"] == "calm_eda_bvp" for r in out.fired_rules)
    assert out.score is not None
    # explain() style payload remains serialisable
    import json

    json.dumps(out.memberships)


def test_engine_conflicting_signals_drive_uncertainty():
    eng = _tiny_engine()
    out = eng.infer(
        {
            "eda_level": 2.0,        # high
            "bvp_variability": -2.0, # low -> conflict
            "temp_deviation": 2.0,
            "motion": -2.0,
            "quality": 0.9,
        }
    )
    assert out.uncertain_mass > 0
    assert any(r["id"] == "conflicting_eda_bvp" for r in out.fired_rules)


# ------------------------------------------------------------ classifier


def _synthetic_fuzzy_inputs(n_subjects=10, personalize=False):
    raw = make_raw(num_subjects=n_subjects)
    feats = extract_features(segment_windows(raw, C.DEFAULT_WINDOW_SAMPLES))
    return build_fuzzy_inputs(feats, personalize=personalize), feats.label.to_numpy()


def test_classifier_fit_predict_and_threshold_origin():
    X, y = _synthetic_fuzzy_inputs()
    m = FuzzyStressClassifier(threshold_mode="train_fit").fit(X, y)
    assert m.threshold_origin_ == "train_fit"
    assert len(m.thresholds_) == 2 and m.thresholds_[0] < m.thresholds_[1]
    full = m.predict_full(X)
    assert set(["score", "decision", "forced_label", "confidence"]).issubset(
        full.columns
    )
    assert set(full.decision.unique()) <= {"low", "uncertain", "high"}
    assert full.confidence.between(0, 1).all()
    # forced predictions must be usable labels
    assert set(np.unique(m.predict(X))) <= {0, 1}


def test_classifier_fixed_threshold_mode_labels_design_choice():
    X, y = _synthetic_fuzzy_inputs(n_subjects=8)
    m = FuzzyStressClassifier(
        threshold_mode="fixed", fixed_thresholds=(0.35, 0.65)
    ).fit(X, y)
    assert m.threshold_origin_ == "fixed_design_choice"
    assert m.thresholds_ == (0.35, 0.65)


def test_quality_ablation_removes_quality_rules():
    X, y = _synthetic_fuzzy_inputs(n_subjects=8)
    m_on = FuzzyStressClassifier(use_quality=True).fit(X, y)
    m_off = FuzzyStressClassifier(use_quality=False).fit(X, y)
    assert any(r.consequent == "uncertain" for r in m_on.rules)
    assert not any(r.consequent == "uncertain" for r in m_off.rules)
    assert all("quality" not in r.antecedents for r in m_off.rules)


def test_personalization_is_subject_relative():
    raw = make_raw(num_subjects=2)
    # subject 2 has a +5 offset on EDA relative to subject 1
    raw.loc[raw.Subject == 2, "eda"] += 5.0
    feats = extract_features(segment_windows(raw, C.DEFAULT_WINDOW_SAMPLES))
    pers = build_fuzzy_inputs(feats, personalize=True)
    unper = build_fuzzy_inputs(feats, personalize=False)
    for col in ("eda_level",):
        s1_p = pers.loc[feats.subject == 1, col].mean()
        s2_p = pers.loc[feats.subject == 2, col].mean()
        # personalised values are centred per subject -> offsets removed
        assert abs(s1_p - s2_p) < 0.5
        s1_u = unper.loc[feats.subject == 1, col].mean()
        s2_u = unper.loc[feats.subject == 2, col].mean()
        assert s2_u - s1_u > 4.0  # unpersonalised keeps the offset
    assert pers.quality.between(0, 1).all()
