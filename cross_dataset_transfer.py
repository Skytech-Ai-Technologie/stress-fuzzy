"""Cross-dataset transfer: WESAD <-> WorkStress3D in a harmonized feature space.

Both datasets come from the same wrist device (Empatica E4) with EDA and
TEMP sampled at 4 Hz and an accelerometer. The harmonized feature vector uses
only channels with identical semantics and rates in both datasets:

    eda__mean, eda__std, eda__slope   (microsiemens, OLS slope per sample)
    temp__mean, temp__std             (degrees Celsius)
    motion (ACC-magnitude std)        (movement proxy)

BVP is excluded: WorkStress3D's copy was downsampled to 4 Hz, so amplitude
statistics are not comparable with 64 Hz BVP. Labels: binary stress (1) vs
non-stress (0); WESAD amusement counts as non-stress, WorkStress3D label 2
(undocumented) is dropped.

Directions (no target-domain data is ever used for fitting):
  A. train WESAD -> test WorkStress3D (pooled + per-participant)
  B. train WorkStress3D -> test WESAD (pooled + per-subject)
Controls: in-dataset LOSO with the SAME harmonized features, so the transfer
decrement is measured against a like-for-like reference.

Run: python cross_dataset_transfer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from stress_fuzzy import config as C
from stress_fuzzy.data import load_physiological, segment_windows
from stress_fuzzy.evaluation import check_leakage, compute_metrics, make_folds
from stress_fuzzy.features import extract_features
from wesad_benchmark import subject_windows

WESAD_DIR = Path(
    "data/kagglehub/datasets/orvile/"
    "wesad-wearable-stress-affect-detection-dataset/versions/1/WESAD"
)
OUT_DIR = Path("bench/transfer")

HARMONIZED = [
    "eda__mean", "eda__std", "eda__slope",
    "temp__mean", "temp__std",
    "motion",
]


def load_wesad() -> pd.DataFrame:
    frames = []
    for d in sorted(p for p in WESAD_DIR.iterdir()
                    if p.is_dir() and p.name.startswith("S")):
        try:
            frames.append(subject_windows(d, "binary"))
        except Exception as exc:
            print(f"  [warn] {d.name}: {exc}")
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"acc_motion": "motion"})
    df["dataset"] = "WESAD"
    return df


def load_workstress() -> pd.DataFrame:
    raw = load_physiological("database", "15sn")
    feats = extract_features(segment_windows(raw, 56, (2,)))
    feats = feats.rename(columns={"acc_magnitude_std": "motion"})
    feats["dataset"] = "WorkStress3D"
    return feats


def models(seed: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=seed)),
        ]),
        "svm_rbf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced",
                        random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300,
                                           min_samples_leaf=2,
                                           class_weight="balanced",
                                           random_state=seed, n_jobs=-1)),
        ]),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] loading WESAD windows ...")
    wes = load_wesad()
    print(f"      {len(wes)} windows, {wes.subject.nunique()} subjects, "
          f"labels {wes.label.value_counts().to_dict()}")
    print("[2/4] loading WorkStress3D windows ...")
    wrk = load_workstress()
    print(f"      {len(wrk)} windows, {wrk.subject.nunique()} participants, "
          f"labels {wrk.label.value_counts().to_dict()}")

    results: dict = {"harmonized_features": HARMONIZED, "n": {}}

    # ---- controls: in-dataset LOSO with the same harmonized features ----
    print("[3/4] in-dataset LOSO controls (harmonized features) ...")
    for name, df in (("wesad", wes), ("workstress3d", wrk)):
        folds = make_folds(df.subject, protocol="loso")
        per_model = {}
        for mname, m in models().items():
            y_true, y_pred = [], []
            for f in folds:
                tr, te = f["train_idx"], f["test_idx"]
                mm = clone(m).fit(df[HARMONIZED].iloc[tr], df.label.to_numpy()[tr])
                y_true.append(df.label.to_numpy()[te])
                y_pred.append(mm.predict(df[HARMONIZED].iloc[te]))
            mets = compute_metrics(np.concatenate(y_true), np.concatenate(y_pred))
            per_model[mname] = {k: mets[k] for k in
                                ("balanced_accuracy", "macro_f1", "n")}
            print(f"  control {name:14s} {mname:20s} BA {mets['balanced_accuracy']:.3f}")
        results["in_dataset_control"] = results.get("in_dataset_control", {})
        results["in_dataset_control"][name] = per_model

    # ---- direction A: train WESAD -> test WorkStress3D ----
    print("[4/4] transfer A: WESAD -> WorkStress3D ...")
    dirA = {}
    for mname, m in models().items():
        m = clone(m).fit(wes[HARMONIZED], wes.label)
        pred = m.predict(wrk[HARMONIZED])
        mets = compute_metrics(wrk.label.to_numpy(), pred)
        dirA[mname] = {k: mets[k] for k in ("balanced_accuracy", "macro_f1",
                                            "confusion_matrix", "n")}
        print(f"  A {mname:20s} BA {mets['balanced_accuracy']:.3f}")
    results["transfer_A_wesad_to_workstress3d"] = dirA

    # ---- direction B: train WorkStress3D -> test WESAD ----
    print("      transfer B: WorkStress3D -> WESAD ...")
    dirB = {}
    for mname, m in models().items():
        m = clone(m).fit(wrk[HARMONIZED], wrk.label)
        pred = m.predict(wes[HARMONIZED])
        mets = compute_metrics(wes.label.to_numpy(), pred)
        dirB[mname] = {k: mets[k] for k in ("balanced_accuracy", "macro_f1",
                                            "confusion_matrix", "n")}
        print(f"  B {mname:20s} BA {mets['balanced_accuracy']:.3f}")
    results["transfer_B_workstress3d_to_wesad"] = dirB

    # per-participant breakdown for direction A (where does transfer survive?)
    m = clone(models()["svm_rbf"]).fit(wes[HARMONIZED], wes.label)
    wrk = wrk.copy()
    wrk["pred"] = m.predict(wrk[HARMONIZED])
    per_part = []
    from sklearn.metrics import balanced_accuracy_score
    for s, g in wrk.groupby("subject"):
        ba = (balanced_accuracy_score(g.label, g.pred)
              if g.label.nunique() > 1 else float("nan"))
        per_part.append({"subject": int(s), "n": int(len(g)),
                         "balanced_accuracy": None if pd.isna(ba) else float(ba),
                         "stress_rate_true": float(g.label.mean()),
                         "stress_rate_pred": float(g.pred.mean())})
    results["transfer_A_per_participant"] = per_part

    (OUT_DIR / "transfer_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[ok] written to {OUT_DIR}/transfer_results.json")
    print("Exploratory research output - not a diagnostic tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
