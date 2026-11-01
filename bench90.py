"""What would it take to reach ~90% on WorkStress3D? (diagnostic study)

Three analyses, all on the real database:

  1. Phase-level classification: aggregate features over whole label phases
     (coarsest honest granularity) and classify phases under LOSO.
  2. Coverage-accuracy: how high does accuracy go if the model answers only
     on its most confident fraction of windows?
  3. The leakage ladder: the same classifier under increasingly leaky
     evaluation choices that appear in published wearables papers -
     overlapping windows with random splits, scalers fit on all data,
     thresholds tuned on test. Quantifies how ~0.90 numbers are produced.

Run: python bench90.py --data-dir database --out-dir bench/bench90
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

from stress_fuzzy import config as C
from stress_fuzzy.baselines import make_baselines
from stress_fuzzy.data import load_physiological, segment_windows
from stress_fuzzy.evaluation import compute_metrics
from stress_fuzzy.features import FEATURE_COLUMNS, extract_features


def base_features(data_dir: Path) -> pd.DataFrame:
    raw = load_physiological(data_dir, "15sn")
    feats = extract_features(segment_windows(raw, 56, (2,)))
    return feats.sort_values(["subject", "phase_id", "window"]).reset_index(drop=True)


# ------------------------------------------------------------------ 1. phase


def phase_level(feats: pd.DataFrame, seed: int = 42) -> dict:
    agg_cols = [c for c in FEATURE_COLUMNS if not c.startswith("quality__")]
    g = feats.groupby(["subject", "phase_id"])
    ph = g[agg_cols].mean().reset_index()
    ph["label"] = g["label"].first().values
    ph["n_windows"] = g.size().values
    X, y = ph[agg_cols], ph.label.to_numpy().astype(int)
    subjects = ph.subject.to_numpy()
    out = {}
    for name, model in make_baselines(seed).items():
        preds = np.zeros_like(y)
        trues = np.zeros_like(y)
        for s in sorted(set(subjects)):
            te = subjects == s
            m = clone(model).fit(X[~te], y[~te])
            preds[te] = m.predict(X[te])
            trues[te] = y[te]
        m = compute_metrics(trues, preds)
        out[name] = {
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "n_phases": int(len(ph)),
            "class_counts": {str(k): int(v) for k, v in
                             pd.Series(trues).value_counts().sort_index().items()},
        }
    return out


# ------------------------------------------------------- 2. coverage-accuracy


def coverage_accuracy(feats: pd.DataFrame, seed: int = 42) -> dict:
    """Sort held-out windows by model confidence; accuracy of top fractions."""
    from sklearn.metrics import balanced_accuracy_score

    X = feats[FEATURE_COLUMNS]
    y = feats.label.to_numpy().astype(int)
    subjects = feats.subject.to_numpy()
    model = make_baselines(seed)["svm_rbf"]

    prob = np.zeros(len(y))
    for s in sorted(set(subjects)):
        te = subjects == s
        m = clone(model).fit(X[~te], y[~te])
        prob[te] = m.predict_proba(X[te])[:, 1]

    conf = np.abs(prob - 0.5) * 2  # 0 = least confident, 1 = most
    order = np.argsort(-conf)
    out = {}
    for cov in (1.0, 0.75, 0.5, 0.25, 0.10, 0.05):
        keep = order[: max(2, int(cov * len(y)))]
        out[f"top_{int(cov*100)}pct"] = {
            "coverage": float(cov),
            "n": int(len(keep)),
            "balanced_accuracy": float(balanced_accuracy_score(y[keep],
                                                               (prob[keep] >= 0.5).astype(int))),
        }
    return out


# --------------------------------------------------------- 3. leakage ladder


def overlapping_windows(raw: pd.DataFrame, win: int, stride: int) -> pd.DataFrame:
    """Windows with overlap; labels from window majority (common published
    practice). Per-subject slicing only - never across participant rows."""
    rows = []
    for subj, grp in raw.groupby("subject"):
        vals = grp[["eda", "bvp", "temp", "x", "y", "z"]].to_numpy(float)
        labels = grp["emotion"].to_numpy()
        for start in range(0, len(grp) - win + 1, stride):
            lab = int(pd.Series(labels[start : start + win]).mode()[0])
            rows.append({"subject": int(subj), "start": start, "end": start + win,
                         "label": lab, "_row": grp.index[0]})
    return pd.DataFrame(rows)


def leakage_ladder(raw: pd.DataFrame, seed: int = 42) -> dict:
    """Balanced accuracy of SVM under progressively leakier protocols."""
    out = {}

    def run_windows(win: int, stride: int, scale_on_all: bool, tag: str):
        from stress_fuzzy.features import channel_stats

        widx_tbl = overlapping_windows(raw, win, stride)
        feats = []
        for r in widx_tbl.itertuples():
            grp = raw[(raw.subject == r.subject)]
            grp = grp.reset_index(drop=True)
            s = grp[["eda", "bvp", "temp", "x", "y", "z"]].to_numpy(float)[r.start : r.end]
            row = {"subject": r.subject, "label": r.label}
            for j, col in enumerate(["eda", "bvp", "temp", "x", "y", "z"]):
                for stat, v in channel_stats(s[:, j]).items():
                    row[f"{col}__{stat}"] = v
            feats.append(row)
        F = pd.DataFrame(feats)
        X = F[[c for c in F.columns if "__" in c]]
        y = F.label.to_numpy().astype(int)
        # exclude degenerate class-2-majority windows from the 2-class metric
        keep = y != 2
        X, y = X[keep].reset_index(drop=True), y[keep]
        model = make_baselines(seed)["svm_rbf"]
        idx_tr, idx_te = train_test_split(
            np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed
        )
        if scale_on_all:
            from sklearn.preprocessing import StandardScaler

            sc = StandardScaler().fit(X)
            Xs = pd.DataFrame(sc.transform(X), columns=X.columns)
            m = clone(model).fit(Xs.iloc[idx_tr], y[idx_tr])
            pred = m.predict(Xs.iloc[idx_te])
        else:
            m = clone(model).fit(X.iloc[idx_tr], y[idx_tr])
            pred = m.predict(X.iloc[idx_te])
        mets = compute_metrics(y[idx_te], pred)
        out[tag] = {
            "n_windows": int(len(y)),
            "balanced_accuracy": mets["balanced_accuracy"],
            "macro_f1": mets["macro_f1"],
        }
        print(f"  {tag:44s} BA {mets['balanced_accuracy']:.3f}  (n={len(y)})")

    print("[ladder] non-overlapping windows, random 75/25 split (mild leak):")
    run_windows(56, 56, False, "L1_no_overlap_random_split")
    print("[ladder] 50% overlap, random split:")
    run_windows(56, 28, False, "L2_overlap50_random_split")
    print("[ladder] 87.5% overlap, random split:")
    run_windows(56, 7, False, "L3_overlap87_random_split")
    print("[ladder] 87.5% overlap + scaler fit on ALL data:")
    run_windows(56, 7, True, "L4_overlap87_scaler_on_all")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="database")
    ap.add_argument("--out-dir", default="bench/bench90")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feats = base_features(Path(args.data_dir))
    raw = load_physiological(Path(args.data_dir), "15sn")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] phase-level LOSO classification ...")
    phase = phase_level(feats, args.seed)
    for m, v in phase.items():
        print(f"  {m:20s} BA {v['balanced_accuracy']:.3f} over {v['n_phases']} phases")

    print("[2/3] coverage-accuracy (LOSO SVM) ...")
    cov = coverage_accuracy(feats, args.seed)
    for k, v in cov.items():
        print(f"  {k:12s} BA {v['balanced_accuracy']:.3f} at {v['coverage']:.2f} coverage (n={v['n']})")

    print("[3/3] leakage ladder ...")
    ladder = leakage_ladder(raw, args.seed)

    (out_dir / "bench90_results.json").write_text(
        json.dumps({"phase_level": phase, "coverage_accuracy": cov,
                    "leakage_ladder": ladder}, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] written to {out_dir}/bench90_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
