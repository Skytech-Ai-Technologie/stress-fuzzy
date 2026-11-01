"""Grouped evaluation: Leave-One-Subject-Out (primary) and GroupKFold (dev).

Hard guarantees:
- Every fold is checked for participant leakage (train/test subject sets must
  be disjoint) before anything is fitted; violations raise ``LeakageError``.
- All fitting (scaling, imputation, fuzzy membership bounds, thresholds,
  class weighting) happens on the training partition of the fold only.
- Oversampling is not used at all; class imbalance is handled with
  class_weight="balanced" fitted on training labels only.

The fuzzy model additionally reports a three-way decision (low/uncertain/high)
so coverage-vs-performance can be measured; baselines have no abstention and
their coverage is 1.0 by definition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold

from . import config as C
from .baselines import make_baselines
from .features import FEATURE_COLUMNS
from .fuzzy_model import FuzzyStressClassifier, build_fuzzy_inputs


class LeakageError(RuntimeError):
    """Raised when a fold shares participants between train and test."""


def check_leakage(train_subjects, test_subjects) -> None:
    overlap = set(train_subjects) & set(test_subjects)
    if overlap:
        raise LeakageError(
            f"participant leakage detected: {sorted(overlap)} appear in both "
            "train and test partitions of a fold"
        )


def make_folds(subjects: pd.Series, protocol: str = "loso", seed: int = 42,
               n_splits: int = 5) -> list[dict]:
    """Return folds as dicts {name, test_subjects, train_idx, test_idx}.

    Indices are positional (0..n-1) into the feature table.
    """
    subjects = pd.Series(subjects).reset_index(drop=True)
    folds = []
    if protocol == "loso":
        for subj in sorted(subjects.unique()):
            test_idx = np.flatnonzero(subjects.values == subj)
            train_idx = np.flatnonzero(subjects.values != subj)
            folds.append(
                {
                    "name": f"subject_{subj}",
                    "test_subjects": [int(subj)],
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
    elif protocol in ("gk", "gk5", "groupkfold"):
        gkf = GroupKFold(n_splits=n_splits)
        for i, (train_idx, test_idx) in enumerate(
            gkf.split(subjects.values, groups=subjects.values)
        ):
            folds.append(
                {
                    "name": f"gkfold_{i}",
                    "test_subjects": sorted(
                        set(subjects.values[test_idx].tolist())
                    ),
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
    else:
        raise ValueError(f"unknown protocol {protocol!r} (use 'loso' or 'gk5')")
    for f in folds:
        check_leakage(
            subjects.values[f["train_idx"]], subjects.values[f["test_idx"]]
        )
    return folds


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Primary metric block; accuracy is reported but never primary."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))
    m = {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }
    p, r, f1c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    m["per_class"] = {
        C.LABEL_NAMES.get(c, str(c)): {
            "precision": float(p[i]), "recall": float(r[i]), "support": int(sup[i])
        }
        for i, c in enumerate(classes)
    }
    m["stress_recall"] = (
        float(r[classes.index(1)]) if 1 in classes else None
    )
    if (
        y_score is not None
        and len(np.unique(y_true)) == 2
        and len(np.unique(y_pred)) <= 2
        and np.ndim(y_score) == 1
    ):
        m["pr_auc"] = float(average_precision_score(y_true, np.asarray(y_score)))
    else:
        m["pr_auc"] = None
    return m


def _participant_metrics(subjects, y_true, y_pred) -> list[dict]:
    rows = []
    for subj in np.unique(subjects):
        m = subjects == subj
        rows.append(
            {
                "subject": int(subj),
                "n_windows": int(m.sum()),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true[m], y_pred[m])
                )
                if len(np.unique(y_true[m])) > 1
                else float("nan"),
            }
        )
    return rows


def _evaluate_fuzzy_on_fold(model, Xtr, ytr, Xte) -> dict:
    model.fit(Xtr, ytr)
    full = model.predict_full(Xte)
    decisions = full["decision"].to_numpy()
    forced = full["forced_label"].to_numpy().astype(int)
    scores = full["score"].fillna(0.5).to_numpy(dtype=float)
    confident = decisions != "uncertain"
    labels = np.where(decisions == "high", 1, np.where(decisions == "low", 0, -1))
    covered_pred = labels[confident]
    return {
        "forced_pred": forced,
        "score": scores,
        "decisions": decisions,
        "covered_mask": confident,
        "covered_pred": covered_pred,
        "coverage": float(confident.mean()) if len(confident) else 0.0,
        "bounds": dict(model.bounds_),
        "thresholds": list(model.thresholds_),
        "threshold_origin": model.threshold_origin_,
    }


def run_experiment(features: pd.DataFrame, cfg: C.Config) -> dict:
    """Run fuzzy variants + baselines over grouped folds.

    Returns a JSON-serialisable results dict with per-fold, per-participant,
    and aggregate metrics plus per-fold fuzzy membership bounds.
    """
    features = features.reset_index(drop=True)
    subjects = features["subject"].to_numpy()
    y = features["label"].to_numpy().astype(int)
    X_base = features[FEATURE_COLUMNS].copy()

    folds = make_folds(
        features["subject"], cfg.protocol, cfg.seed, cfg.group_kfold_splits
    )

    # Fuzzy input variants (personalisation is an input-construction choice).
    fuzzy_inputs = {
        "fuzzy_quality_personalized": build_fuzzy_inputs(features, personalize=True),
    }
    if cfg.personalize:
        fuzzy_inputs["fuzzy_quality_unpersonalized"] = build_fuzzy_inputs(
            features, personalize=False
        )
    if cfg.quality_aware:
        fuzzy_inputs["fuzzy_no_quality"] = build_fuzzy_inputs(
            features, personalize=cfg.personalize
        )

    def make_fuzzy(name: str):
        if name == "fuzzy_no_quality":
            return FuzzyStressClassifier(
                rules_path=cfg.rules_path or None,
                use_quality=False,
                threshold_mode=cfg.threshold_mode,
                fixed_thresholds=cfg.fixed_thresholds,
                confidence_tau=cfg.confidence_tau,
            )
        return FuzzyStressClassifier(
            rules_path=cfg.rules_path or None,
            use_quality=True,
            threshold_mode=cfg.threshold_mode,
            fixed_thresholds=cfg.fixed_thresholds,
            confidence_tau=cfg.confidence_tau,
        )

    model_specs: dict = {}
    for fname in fuzzy_inputs:
        model_specs[fname] = ("fuzzy", make_fuzzy(fname))
    for bname, bpipe in make_baselines(cfg.seed).items():
        model_specs[bname] = ("baseline", bpipe)

    results = {
        "protocol": cfg.protocol,
        "n_participants": int(len(np.unique(subjects))),
        "n_windows": int(len(features)),
        "class_counts": {str(k): int(v) for k, v in
                         pd.Series(y).value_counts().sort_index().items()},
        "folds": [f["name"] for f in folds],
        "models": {},
        "fold_fuzzy_bounds": {},
        "fold_thresholds": {},
        "notes": [
            "LOSO: each participant is held out once; no windows of the held-out "
            "participant enter training or any fit step",
            "fuzzy forced_label maps uncertain to the 0.5 score split for "
            "baseline comparability; coverage metrics quantify abstention",
            f"personalisation={'on' if cfg.personalize else 'off'}: per-participant "
            "robust z-scoring of fuzzy inputs using that participant's own windows "
            "only (no labels used)",
        ],
    }

    for name, (kind, prototype) in model_specs.items():
        fold_records = []
        part_records = []
        y_true_all, y_pred_all, y_score_all = [], [], []
        y_true_cov, y_pred_cov = [], []
        for fold in folds:
            tr, te = fold["train_idx"], fold["test_idx"]
            check_leakage(subjects[tr], subjects[te])
            if kind == "fuzzy":
                X = fuzzy_inputs[name]
                model = clone_fuzzy(prototype)
                info = _evaluate_fuzzy_on_fold(model, X.iloc[tr], y[tr], X.iloc[te])
                pred, score = info["forced_pred"], info["score"]
                if name not in results["fold_fuzzy_bounds"]:
                    results["fold_fuzzy_bounds"][name] = {}
                results["fold_fuzzy_bounds"][name][fold["name"]] = info["bounds"]
                if name not in results["fold_thresholds"]:
                    results["fold_thresholds"][name] = {}
                results["fold_thresholds"][name][fold["name"]] = {
                    "thresholds": info["thresholds"],
                    "origin": info["threshold_origin"],
                }
                cov_mask = info["covered_mask"]
                y_true_cov.extend(y[te][cov_mask].tolist())
                y_pred_cov.extend(info["covered_pred"].tolist())
                fold_records.append(
                    {
                        "fold": fold["name"],
                        "test_subjects": fold["test_subjects"],
                        **_metrics_with_coverage(
                            y[te], pred, score, info["coverage"]
                        ),
                        "decisions": {
                            d: int((info["decisions"] == d).sum())
                            for d in ("low", "uncertain", "high")
                        },
                    }
                )
            else:
                model = clone(prototype)
                model.fit(X_base.iloc[tr], y[tr])
                pred = model.predict(X_base.iloc[te]).astype(int)
                try:
                    score = model.predict_proba(X_base.iloc[te])[:, 1]
                except Exception:
                    score = None
                fold_records.append(
                    {
                        "fold": fold["name"],
                        "test_subjects": fold["test_subjects"],
                        **_metrics_with_coverage(y[te], pred, score, 1.0),
                        "decisions": None,
                    }
                )
            y_true_all.extend(y[te].tolist())
            y_pred_all.extend(np.asarray(pred).tolist())
            if score is not None:
                y_score_all.extend(np.asarray(score).tolist())
            part_records.extend(_participant_metrics(subjects[te], y[te], np.asarray(pred)))

        agg = _aggregate(fold_records)
        model_result = {
            "kind": kind,
            "fold_metrics": fold_records,
            "participant_metrics": part_records,
            "aggregate": agg,
            "overall_metrics": compute_metrics(
                y_true_all, y_pred_all, np.array(y_score_all) if y_score_all else None
            ),
        }
        if kind == "fuzzy" and y_true_cov:
            model_result["covered_only_metrics"] = compute_metrics(
                y_true_cov, y_pred_cov
            )
        results["models"][name] = model_result

    results["comparisons"] = _compare_to_fuzzy(results)
    return results


def clone_fuzzy(model: FuzzyStressClassifier) -> FuzzyStressClassifier:
    """Fresh unfitted copy with identical hyper-parameters."""
    return FuzzyStressClassifier(
        rules_path=model.rules_path,
        use_quality=model.use_quality,
        threshold_mode=model.threshold_mode,
        fixed_thresholds=model.fixed_thresholds,
        confidence_tau=model.confidence_tau,
    )


def _metrics_with_coverage(y_true, y_pred, y_score, coverage) -> dict:
    m = compute_metrics(y_true, y_pred, y_score)
    m["coverage"] = float(coverage)
    return m


def _aggregate(fold_records: list[dict]) -> dict:
    keys = [
        "macro_f1",
        "balanced_accuracy",
        "stress_recall",
        "pr_auc",
        "coverage",
        "accuracy",
    ]
    agg = {}
    for k in keys:
        vals = [r[k] for r in fold_records if r.get(k) is not None]
        if not vals:
            agg[k] = None
            continue
        arr = np.asarray(vals, dtype=float)
        agg[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_folds": int(arr.size),
        }
    agg["n_windows_total"] = int(sum(r["n"] for r in fold_records))
    return agg


def _compare_to_fuzzy(results: dict) -> dict:
    """Wilcoxon signed-rank on per-participant balanced accuracy (exploratory)."""
    from scipy.stats import wilcoxon

    fuzzy_name = "fuzzy_quality_personalized"
    if fuzzy_name not in results["models"]:
        return {}
    ref = {
        r["subject"]: r["balanced_accuracy"]
        for r in results["models"][fuzzy_name]["participant_metrics"]
    }
    comps = {}
    for name, mres in results["models"].items():
        if name == fuzzy_name:
            continue
        cur = {
            r["subject"]: r["balanced_accuracy"]
            for r in mres["participant_metrics"]
        }
        shared = sorted(set(ref) & set(cur))
        pairs = [
            (ref[s], cur[s]) for s in shared
            if np.isfinite(ref[s]) and np.isfinite(cur[s])
        ]
        if len(pairs) < 6:
            comps[name] = {"n_pairs": len(pairs), "test": "not enough pairs (min 6)"}
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        try:
            if np.allclose(a - b, 0.0):
                comps[name] = {
                    "n_pairs": len(pairs),
                    "test": "not computable: identical per-participant metrics",
                }
                continue
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # degenerate pairs
                stat, p = wilcoxon(a, b, zero_method="wilcox")
            comps[name] = {
                "n_pairs": len(pairs),
                "median_diff": float(np.median(a - b)),
                "wilcoxon_stat": float(stat),
                "p_value": float(p),
                "interpretation": "exploratory only; not a confirmatory test",
            }
        except ValueError as exc:
            comps[name] = {"n_pairs": len(pairs), "test": f"not computable: {exc}"}
    return comps
