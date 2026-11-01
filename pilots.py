"""Pilot experiments for contribution candidates (exploratory).

Run:  python pilots.py --data-dir database --out-dir bench/pilots

Pilots (all reuse the grouped-fold machinery from stress_fuzzy):
  1. Personal few-shot calibration: how many labeled windows from the HELD-OUT
     participant are needed to lift performance above the LOSO cold start?
     Calibration windows are sampled from the test participant only, the model
     is refit on train participants + calibration windows, and evaluation uses
     only the participant's REMAINING windows (never the calibration ones).
  2. Leakage inflation: identical models under random-window KFold (leaks
     adjacent windows of the same participant and phase), phase-grouped KFold
     (same phase together, participant still leaks), GroupKFold by participant,
     and LOSO - quantifying how evaluation protocol choice inflates results.
  3. Protocol-aware temporal smoothing: 2-state Viterbi over each held-out
     participant's ordered score sequence, with persistence (transition)
     estimated from TRAINING participants' label sequences only. Uses no label
     of the test participant.

These are exploratory analyses for research-direction decisions, not
confirmatory results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold

from stress_fuzzy import config as C
from stress_fuzzy.baselines import make_baselines
from stress_fuzzy.data import load_physiological, segment_windows
from stress_fuzzy.evaluation import compute_metrics, make_folds
from stress_fuzzy.features import FEATURE_COLUMNS, extract_features
from stress_fuzzy.fuzzy_model import FuzzyStressClassifier, build_fuzzy_inputs


def build_features(data_dir: Path, cfg: C.Config) -> pd.DataFrame:
    raw = load_physiological(data_dir, cfg.window_file)
    natural = 56  # base unit of the 15sn file; pilot uses 15s windows
    widx = segment_windows(raw, natural, tuple(cfg.drop_labels))
    feats = extract_features(widx)
    return feats.sort_values(["subject", "phase_id", "window"]).reset_index(drop=True)


# ---------------------------------------------------------------- pilot 1


def pilot_calibration(feats: pd.DataFrame, seed: int = 42,
                      k_grid=(0, 1, 2, 5, 10, 20), reps: int = 20) -> dict:
    """Few-shot personal calibration curves with the SVM baseline."""
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(seed)
    subjects = sorted(feats.subject.unique())
    base = make_baselines(seed)["svm_rbf"]
    records = []

    for subj in subjects:
        te = feats[feats.subject == subj]
        tr = feats[feats.subject != subj]
        for k in k_grid:
            if k == 0:
                model = clone(base).fit(tr[FEATURE_COLUMNS], tr.label)
                pred = model.predict(te[FEATURE_COLUMNS])
                records.append(
                    {"subject": int(subj), "k": 0, "rep": 0,
                     "balanced_accuracy": float(
                         balanced_accuracy_score(te.label, pred)),
                     "n_eval": int(len(te))}
                )
                continue
            for rep in range(reps):
                cal_idx = _stratified_sample(te, k, rng)
                if cal_idx is None:
                    continue
                cal = te.loc[cal_idx]
                ev = te.drop(cal_idx)
                if ev.label.nunique() < 2 or len(ev) < 4:
                    continue
                model = clone(base)
                aug = pd.concat([tr[FEATURE_COLUMNS + ["label"]], cal], ignore_index=True)
                model.fit(aug[FEATURE_COLUMNS], aug.label)
                pred = model.predict(ev[FEATURE_COLUMNS])
                records.append(
                    {"subject": int(subj), "k": k, "rep": rep,
                     "balanced_accuracy": float(
                         balanced_accuracy_score(ev.label, pred)),
                     "n_eval": int(len(ev))}
                )

    df = pd.DataFrame(records)
    summary = {}
    for k, grp in df.groupby("k"):
        per_subj = grp.groupby("subject").balanced_accuracy.mean()
        summary[int(k)] = {
            "mean_balanced_accuracy": float(per_subj.mean()),
            "std_across_participants": float(per_subj.std()),
            "n_participants": int(per_subj.size),
            "participants_ge_0.7": int((per_subj >= 0.7).sum()),
            "total_fits": int(len(grp)),
        }
    return {"records": df.to_dict("records"), "summary": summary}


def _stratified_sample(te: pd.DataFrame, k: int, rng) -> list | None:
    """Sample k window indices stratified by label; both classes if possible."""
    labels = te.label.to_numpy()
    idx = np.arange(len(te))
    picks: list = []
    per_class = {c: idx[labels == c] for c in (0, 1)}
    if any(len(v) == 0 for v in per_class.values()):
        return None
    half = k // 2
    for c, take in ((0, k - half), (1, half)):
        pool = per_class[c]
        take = min(take, len(pool))
        picks.extend(rng.choice(pool, size=take, replace=False).tolist())
    return te.index[picks].tolist()


def pilot_calibration_threshold(feats: pd.DataFrame, seed: int = 42,
                                k_grid=(2, 5, 10, 20), reps: int = 20) -> dict:
    """Personal THRESHOLD calibration: LOSO model frozen, only the decision
    threshold is re-chosen on k labeled windows of the test participant."""
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(seed + 1)
    X = feats[FEATURE_COLUMNS]
    y = feats.label.to_numpy().astype(int)
    base = make_baselines(seed)["svm_rbf"]
    records = []

    for subj in sorted(feats.subject.unique()):
        te = feats.index[feats.subject == subj]
        tr = feats.index[feats.subject != subj]
        model = clone(base).fit(X.loc[tr], y[tr])
        prob_te = model.predict_proba(X.loc[te])[:, 1]
        te = te.to_numpy()
        for k in k_grid:
            for rep in range(reps):
                local = np.arange(len(te))
                labels_te = feats.loc[te, "label"].to_numpy()
                picks: list = []
                for c, take in ((0, k - k // 2), (1, k // 2)):
                    pool = local[labels_te == c]
                    if len(pool) == 0:
                        picks = []
                        break
                    picks.extend(rng.choice(pool, size=min(take, len(pool)),
                                            replace=False).tolist())
                if not picks:
                    continue
                cal_p, cal_y = prob_te[picks], labels_te[picks]
                if len(np.unique(cal_y)) < 2:
                    continue
                ev = np.setdiff1d(local, picks)
                if len(np.unique(labels_te[ev])) < 2 or len(ev) < 4:
                    continue
                best_t, best_ba = 0.5, -1
                for t in np.linspace(0.2, 0.8, 61):
                    ba = balanced_accuracy_score(
                        cal_y, (cal_p >= t).astype(int)
                    )
                    if ba > best_ba:
                        best_ba, best_t = ba, t
                pred = (prob_te[ev] >= best_t).astype(int)
                records.append(
                    {"subject": int(subj), "k": k, "rep": rep,
                     "balanced_accuracy": float(
                         balanced_accuracy_score(labels_te[ev], pred)),
                     "n_eval": int(len(ev))}
                )

    df = pd.DataFrame(records)
    summary = {}
    for k, grp in df.groupby("k"):
        per_subj = grp.groupby("subject").balanced_accuracy.mean()
        summary[int(k)] = {
            "mean_balanced_accuracy": float(per_subj.mean()),
            "std_across_participants": float(per_subj.std()),
            "n_participants": int(per_subj.size),
            "participants_ge_0.7": int((per_subj >= 0.7).sum()),
        }
    return {"records": df.to_dict("records"), "summary": summary}


# ---------------------------------------------------------------- pilot 2


def _fit_eval(model, X_tr, y_tr, X_te, y_te) -> dict:
    model = clone(model).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    try:
        score = model.predict_proba(X_te)[:, 1]
    except Exception:
        score = None
    return compute_metrics(y_te, pred, score)


def pilot_leakage(feats: pd.DataFrame, seed: int = 42) -> dict:
    """Identical models under increasingly leaky protocols."""
    X = feats[FEATURE_COLUMNS]
    y = feats.label.to_numpy().astype(int)

    protocols: dict = {}

    # (a) random window KFold - leaks adjacent windows of same participant/phase
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(kfold.split(X))

    # (b) phase-grouped KFold - same phase stays together, participant leaks
    phase_groups = (feats.subject.astype(str) + "_" + feats.phase_id.astype(str)).to_numpy()
    gk_phase = list(GroupKFold(n_splits=5).split(X, groups=phase_groups))

    # (c) participant-grouped KFold
    gk_subj = list(GroupKFold(n_splits=5).split(X, groups=feats.subject.to_numpy()))

    # (d) LOSO
    loso = [(np.flatnonzero(feats.subject.to_numpy() != s),
             np.flatnonzero(feats.subject.to_numpy() == s))
            for s in sorted(feats.subject.unique())]

    # (e) within-participant 5-fold: train/test windows from the SAME person
    # only - upper bound on what subject-local structure the features carry.
    from sklearn.model_selection import StratifiedKFold

    within = []
    subj_arr = feats.subject.to_numpy()
    for s in sorted(feats.subject.unique()):
        idx = np.flatnonzero(subj_arr == s)
        labels_s = y[idx]
        if min((labels_s == 0).sum(), (labels_s == 1).sum()) < 5:
            continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_local, te_local in skf.split(idx, labels_s):
            within.append((idx[tr_local], idx[te_local]))

    protocols = {
        "random_kfold5_leaky": folds,
        "phase_grouped_kfold5": gk_phase,
        "participant_grouped_kfold5": gk_subj,
        "leave_one_subject_out": loso,
        "within_participant_5fold": within,
    }

    out = {}
    for pname, fl in protocols.items():
        out[pname] = {}
        for mname, model in make_baselines(seed).items():
            per_fold = [_fit_eval(model, X.iloc[tr], y[tr], X.iloc[te], y[te])
                        for tr, te in fl]
            bacc = [m["balanced_accuracy"] for m in per_fold]
            f1 = [m["macro_f1"] for m in per_fold]
            out[pname][mname] = {
                "balanced_accuracy_mean": float(np.mean(bacc)),
                "balanced_accuracy_std": float(np.std(bacc, ddof=1)),
                "macro_f1_mean": float(np.mean(f1)),
                "n_folds": len(fl),
            }
    return out


# ---------------------------------------------------------------- pilot 3


def _viterbi(p: np.ndarray, stay: float) -> np.ndarray:
    """MAP path of a 2-state chain; emission of state 1 is p, of state 0 is 1-p."""
    p = np.clip(p, 1e-3, 1 - 1e-3)
    switch = 1.0 - stay
    log_trans = np.log([[stay, switch], [switch, stay]])
    n = len(p)
    delta = np.zeros(2)
    psi = np.zeros((n, 2), dtype=int)
    delta[0] = np.log(1 - p[0])
    delta[1] = np.log(p[0])
    for t in range(1, n):
        cand = delta[:, None] + log_trans  # [from, to]
        psi[t] = np.argmax(cand, axis=0)
        emit = np.array([np.log(1 - p[t]), np.log(p[t])])
        delta = cand[psi[t], [0, 1]] + emit
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]
    return path


def pilot_smoothing(feats: pd.DataFrame, seed: int = 42) -> dict:
    """Viterbi smoothing of held-out score sequences (SVM probabilities)."""
    from scipy.stats import wilcoxon
    from sklearn.metrics import balanced_accuracy_score

    X = feats[FEATURE_COLUMNS]
    y = feats.label.to_numpy().astype(int)
    subjects = feats.subject.to_numpy()
    base = make_baselines(seed)["svm_rbf"]

    per_subject = []
    for subj in sorted(feats.subject.unique()):
        te = np.flatnonzero(subjects == subj)
        tr = np.flatnonzero(subjects != subj)
        model = clone(base).fit(X.iloc[tr], y[tr])
        prob = model.predict_proba(X.iloc[te])[:, 1]

        # persistence estimated from TRAINING participants' label sequences only
        tr_labels = feats.iloc[tr].sort_values(["subject", "phase_id", "window"]).label.to_numpy()
        tr_subj = feats.iloc[tr].sort_values(["subject", "phase_id", "window"]).subject.to_numpy()
        same_next = (tr_labels[1:] == tr_labels[:-1]) & (tr_subj[1:] == tr_subj[:-1])
        valid = tr_subj[1:] == tr_subj[:-1]
        stay = float(same_next.sum() / valid.sum()) if valid.sum() else 0.8

        raw_pred = (prob >= 0.5).astype(int)
        sm_pred = _viterbi(prob, stay)
        per_subject.append(
            {
                "subject": int(subj),
                "stay_prob": round(stay, 4),
                "raw_balanced_accuracy": float(balanced_accuracy_score(y[te], raw_pred)),
                "smoothed_balanced_accuracy": float(balanced_accuracy_score(y[te], sm_pred)),
                "n_windows": int(len(te)),
            }
        )

    df = pd.DataFrame(per_subject)
    diff = df.smoothed_balanced_accuracy - df.raw_balanced_accuracy
    try:
        stat, p = wilcoxon(df.smoothed_balanced_accuracy, df.raw_balanced_accuracy)
        p = float(p)
    except ValueError:
        p = None
    return {
        "per_subject": per_subject,
        "summary": {
            "raw_mean": float(df.raw_balanced_accuracy.mean()),
            "smoothed_mean": float(df.smoothed_balanced_accuracy.mean()),
            "mean_delta": float(diff.mean()),
            "wilcoxon_p": p,
            "improved_participants": int((diff > 0).sum()),
            "worsened_participants": int((diff < 0).sum()),
            "n_participants": int(len(df)),
        },
    }


# -------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="database")
    ap.add_argument("--out-dir", default="bench/pilots")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = C.Config()
    feats = build_features(Path(args.data_dir), cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pilot 1] personal few-shot calibration (SVM, {feats.subject.nunique()} participants) ...")
    cal = pilot_calibration(feats, seed=args.seed)
    print(json.dumps(cal["summary"], indent=2))

    print("[pilot 1b] personal threshold calibration (frozen LOSO model) ...")
    cal_t = pilot_calibration_threshold(feats, seed=args.seed)
    print(json.dumps(cal_t["summary"], indent=2))

    print("[pilot 2] leakage inflation across protocols ...")
    leak = pilot_leakage(feats, seed=args.seed)
    for proto, models in leak.items():
        line = "  ".join(
            f"{m}: BA {v['balanced_accuracy_mean']:.3f}" for m, v in models.items()
        )
        print(f"  {proto:28s} {line}")

    print("[pilot 3] protocol-aware Viterbi smoothing (SVM probabilities) ...")
    smooth = pilot_smoothing(feats, seed=args.seed)
    print(json.dumps(smooth["summary"], indent=2))

    (out_dir / "calibration_curve.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    (out_dir / "calibration_threshold.json").write_text(json.dumps(cal_t, indent=2), encoding="utf-8")
    (out_dir / "leakage_inflation.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")
    (out_dir / "hmm_smoothing.json").write_text(json.dumps(smooth, indent=2), encoding="utf-8")

    lines = ["# Contribution pilots (exploratory)", ""]
    lines.append("## 1. Personal few-shot calibration (SVM, balanced accuracy)")
    lines.append("")
    lines.append("| calibration windows k | mean BA | ± std (participants) | participants ≥ 0.70 |")
    lines.append("|---|---|---|---|")
    for k, s in cal["summary"].items():
        lines.append(f"| {k} | {s['mean_balanced_accuracy']:.3f} | {s['std_across_participants']:.3f} | {s['participants_ge_0.7']}/{s['n_participants']} |")
    lines.append("")
    lines.append("## 1b. Personal threshold calibration (frozen LOSO model)")
    lines.append("")
    lines.append("| calibration windows k | mean BA | ± std | participants ≥ 0.70 |")
    lines.append("|---|---|---|---|")
    for k, s in cal_t["summary"].items():
        lines.append(f"| {k} | {s['mean_balanced_accuracy']:.3f} | {s['std_across_participants']:.3f} | {s['participants_ge_0.7']}/{s['n_participants']} |")
    lines.append("")
    lines.append("## 2. Leakage inflation (balanced accuracy, same models)")
    lines.append("")
    lines.append("| protocol | logreg | SVM | RF |")
    lines.append("|---|---|---|---|")
    for proto, models in leak.items():
        lines.append(
            f"| {proto} | "
            + " | ".join(f"{models[m]['balanced_accuracy_mean']:.3f}" for m in
                         ("logistic_regression", "svm_rbf", "random_forest"))
            + " |"
        )
    lines.append("")
    s = smooth["summary"]
    lines.append("## 3. Protocol-aware Viterbi smoothing (SVM)")
    lines.append("")
    lines.append(f"- raw BA {s['raw_mean']:.3f} -> smoothed BA {s['smoothed_mean']:.3f} "
                 f"(mean delta {s['mean_delta']:+.3f}, Wilcoxon p={s['wilcoxon_p']})")
    lines.append(f"- improved {s['improved_participants']}/{s['n_participants']} participants, "
                 f"worsened {s['worsened_participants']}")
    lines.append("")
    (out_dir / "pilot_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] pilots written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
