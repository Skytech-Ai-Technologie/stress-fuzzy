"""WESAD loader + LOSO benchmark for the fuzzy stress model.

Data: Schmidt et al. 2018 (ICMI), 15 subjects, Empatica E4 wrist (BVP 64 Hz,
EDA/TEMP 4 Hz, ACC 32 Hz) + chest reference. Labels (700 Hz track):
1=baseline, 2=stress, 3=amusement, 4=meditation, 0/5/6/7=ignore.

Task (configurable):
  binary  : stress (2) vs non-stress (baseline 1 + amusement 3)
  3class  : baseline vs stress vs amusement

Protocol: non-overlapping fixed windows within contiguous same-label phases
(same construction as the WorkStress3D pipeline), Leave-One-Subject-Out,
fold-local preprocessing. Fuzzy membership bounds + thresholds from training
subjects only. HRV features are computed from 64 Hz BVP (impossible on
WorkStress3D's 4 Hz copy) with fold-local use only.

Run: python wesad_benchmark.py --data-dir data/WESAD --out-dir bench/wesad
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

from stress_fuzzy import config as C
from stress_fuzzy.baselines import make_baselines
from sklearn.metrics import f1_score
from stress_fuzzy.fuzzy_model import FuzzyEngine, load_rules, parse_rules
from stress_fuzzy.evaluation import (
    LeakageError,
    check_leakage,
    compute_metrics,
)
from stress_fuzzy.fuzzy_model import FuzzyStressClassifier
from sklearn.base import clone

EDA_HZ, TEMP_HZ, BVP_HZ, ACC_HZ = 4, 4, 64, 32
WINDOW_SECONDS = 60
LABEL_MAP_BINARY = {1: 0, 3: 0, 2: 1}          # baseline/amusement = non-stress
LABEL_MAP_3CLASS = {1: 0, 2: 1, 3: 2}
VALID_TASKS = ("binary", "3class")

HRV_MIN_PEAKS = 8   # ~8 beats/min floor for valid HRV in a 60 s window


def load_subject(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as fh:
        return pickle.load(fh, encoding="latin1")


def dominant_label(labels: np.ndarray) -> int:
    """Longest-run label in a window (robust to short transients)."""
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[np.argmax(counts)])


def subject_windows(subj_dir: Path, task: str = "binary") -> pd.DataFrame:
    """Cut one subject's wrist streams into non-overlapping 60 s windows.

    Windows keep a single dominant label (longest-run rule); windows whose
    label is not part of the task map are dropped and counted.
    """
    pkl = next(subj_dir.glob("*.pkl"), None)
    if pkl is None:
        raise FileNotFoundError(f"no .pkl in {subj_dir}")
    data = load_subject(pkl)
    sig = data["signal"]["wrist"]
    label700 = data["label"].ravel()
    # mirror-dependent subject field: int in the original release, 'S2' here
    subj = int(str(data["subject"]).lstrip("Ss"))

    n700 = len(label700)
    win700 = WINDOW_SECONDS * 700
    rows = []
    dropped_unknown = 0

    def slice_by_rate(arr: np.ndarray, hz: int, start: int, stop: int) -> np.ndarray:
        lo = int(start / 700 * hz)
        hi = min(int(stop / 700 * hz), len(arr))
        out = arr[:hi][lo:]
        if out.ndim == 2 and out.shape[1] == 1:
            out = out.ravel()
        return out if hi > lo else np.empty(0)

    for start in range(0, n700 - win700 + 1, win700):
        stop = start + win700
        lab = dominant_label(label700[start:stop])
        mapping = LABEL_MAP_BINARY if task == "binary" else LABEL_MAP_3CLASS
        if lab not in mapping:
            dropped_unknown += 1
            continue
        eda = slice_by_rate(sig["EDA"], EDA_HZ, start, stop)
        temp = slice_by_rate(sig["TEMP"], TEMP_HZ, start, stop)
        bvp = slice_by_rate(sig["BVP"], BVP_HZ, start, stop)
        acc = slice_by_rate(sig["ACC"], ACC_HZ, start, stop)
        if len(eda) < EDA_HZ * 10 or len(bvp) < BVP_HZ * 10:
            dropped_unknown += 1
            continue

        row = {
            "subject": subj,
            "label": mapping[lab],
            "phase_label_raw": lab,
            "start": start,
        }
        # interpretable stats
        row["eda__mean"], row["eda__std"] = float(np.mean(eda)), float(np.std(eda))
        row["eda__slope"] = float(np.polyfit(np.arange(len(eda)), eda, 1)[0])
        row["temp__mean"], row["temp__std"] = float(np.mean(temp)), float(np.std(temp))
        row["bvp__std"] = float(np.std(bvp))
        accmag = np.linalg.norm(acc, axis=1)
        row["acc_motion"] = float(np.std(accmag))
        row.update(hrv_features(bvp))
        row.update(eda_quality(eda))
        # chest view (all channels at 700 Hz): EDA, ECG -> HR/HRV, Temp
        chest = data["signal"]["chest"]
        c_eda = slice_by_rate(chest["EDA"], 700, start, stop)
        c_ecg = slice_by_rate(chest["ECG"], 700, start, stop)
        c_temp = slice_by_rate(chest["Temp"], 700, start, stop)
        row["chest__eda_mean"] = float(np.mean(c_eda))
        row["chest__eda_std"] = float(np.std(c_eda))
        row["chest__temp_mean"] = float(np.mean(c_temp))
        row.update({f"chest__{k}": v for k, v in
                    ecg_hrv_features(c_ecg).items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["dropped_unknown_label_windows"] = dropped_unknown
    return df


def hrv_features(bvp: np.ndarray) -> dict:
    """Time-domain HRV from pulse peaks of the 64 Hz BVP signal.

    Returns NaN for windows without enough clean beats rather than inventing
    values; the fuzzy engine treats NaN as missing input (rules skip).
    """
    out = {"hr_mean": np.nan, "hrv_rmssd": np.nan, "hrv_sdnn": np.nan}
    if len(bvp) < BVP_HZ * 10:
        return out
    bvp = bvp - np.median(bvp)
    peaks, props = find_peaks(
        bvp,
        distance=int(0.4 * BVP_HZ),   # max 150 bpm
        prominence=np.std(bvp) * 0.5,
    )
    if len(peaks) < HRV_MIN_PEAKS:
        return out
    ibi = np.diff(peaks) / BVP_HZ  # seconds between beats
    if len(ibi) < HRV_MIN_PEAKS - 1 or np.any(ibi <= 0):
        return out
    hr = 60.0 / ibi
    if not (30 <= np.median(hr) <= 180):  # implausible -> artifact window
        return out
    out["hr_mean"] = float(np.mean(hr))
    out["hrv_rmssd"] = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
    out["hrv_sdnn"] = float(np.std(ibi, ddof=1))
    return out


def ecg_hrv_features(ecg: np.ndarray) -> dict:
    """R-peak based HR/HRV from the 700 Hz chest ECG (cleaner than BVP)."""
    out = {"hr_mean": np.nan, "hrv_rmssd": np.nan, "hrv_sdnn": np.nan}
    if len(ecg) < 700 * 10:
        return out
    ecg = ecg - np.median(ecg)
    peaks, _ = find_peaks(
        ecg, distance=int(0.4 * 700), prominence=float(np.std(ecg)) * 0.6
    )
    if len(peaks) < HRV_MIN_PEAKS:
        return out
    ibi = np.diff(peaks) / 700
    if len(ibi) < HRV_MIN_PEAKS - 1 or np.any(ibi <= 0):
        return out
    hr = 60.0 / ibi
    if not (30 <= np.median(hr) <= 180):
        return out
    out["hr_mean"] = float(np.mean(hr))
    out["hrv_rmssd"] = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
    out["hrv_sdnn"] = float(np.std(ibi, ddof=1))
    return out


def eda_quality(eda: np.ndarray) -> dict:
    """Flatline/missing flags for the EDA channel (quality gate input)."""
    finite = eda[np.isfinite(eda)]
    flat = finite.size < 2 or float(np.std(finite)) < 1e-6
    missing = float(np.mean(~np.isfinite(eda)))
    return {"eda_flat": bool(flat), "eda_missing_frac": missing}


FEATURES = [
    "eda__mean", "eda__std", "eda__slope",
    "temp__mean", "temp__std",
    "bvp__std", "hr_mean", "hrv_rmssd", "hrv_sdnn",
    "acc_motion", "eda_missing_frac",
]

CHEST_FEATURES = [
    "chest__eda_mean", "chest__eda_std", "chest__temp_mean",
    "chest__hr_mean", "chest__hrv_rmssd", "chest__hrv_sdnn",
]


def loso_benchmark(df: pd.DataFrame, task: str, seed: int) -> dict:
    subjects = df.subject.to_numpy()
    y = df.label.to_numpy().astype(int)
    X = df[FEATURES]

    # fuzzy inputs mirror the WorkStress3D design, plus HRV (available here
    # because BVP is 64 Hz). Column names must match the WESAD rules YAML.
    def fuzzy_frame(frame: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=frame.index)
        f["eda_level"] = frame["eda__mean"].astype(float)
        f["bvp_variability"] = frame["bvp__std"].astype(float)
        f["temp_deviation"] = (frame["temp__mean"].astype(float) - 33.0).abs()
        f["motion"] = frame["acc_motion"].astype(float)
        f["hr_mean"] = frame["hr_mean"].astype(float)
        f["hrv_rmssd"] = frame["hrv_rmssd"].astype(float)
        q = 1.0
        q -= 0.35 * frame["eda_missing_frac"].fillna(1.0)
        q -= 0.35 * frame["eda_flat"].astype(float)
        f["quality"] = q.clip(0, 1)
        return f

    import stress_fuzzy as _sf

    WESAD_RULES = Path(_sf.__file__).parent / "rules" / "wesad_rules.yaml"
    fuzzy_specs = {
        "fuzzy_hrv_quality": dict(use_quality=True, personalize=False,
                                  rules=WESAD_RULES),
        "fuzzy_hrv_personalized": dict(use_quality=True, personalize=True,
                                       rules=WESAD_RULES),
        "fuzzy_no_quality": dict(use_quality=False, personalize=False,
                                 rules=WESAD_RULES),
    }

    def personalize(frame: pd.DataFrame) -> pd.DataFrame:
        """Per-subject robust z-score (own windows only; no labels)."""
        out = frame.copy()
        for col in ("eda_level", "bvp_variability", "temp_deviation", "motion",
                    "hr_mean", "hrv_rmssd"):
            v = out[col].to_numpy(float)
            z = np.empty_like(v)
            for s in np.unique(subjects):
                m = df.loc[out.index, "subject"].to_numpy() == s
                med = np.nanmedian(v[m])
                mad = np.nanmedian(np.abs(v[m] - med))
                scale = 1.4826 * mad if mad > 1e-9 else (np.nanstd(v[m]) or 1.0)
                z[m] = (v[m] - med) / scale
            out[col] = z
        return out

    folds = []
    for s in sorted(np.unique(subjects)):
        te = subjects == s
        folds.append((np.flatnonzero(~te), np.flatnonzero(te), int(s)))

    results = {"task": task, "n_subjects": int(len(np.unique(subjects))),
               "n_windows": int(len(df)),
               "class_counts": {str(k): int(v) for k, v in
                                pd.Series(y).value_counts().sort_index().items()},
                "models": {}}

    def run(model_name, fit_predict):
        y_true_all, y_pred_all, score_all = [], [], []
        dec_all = []
        per_fold = []
        for tr, te, s in folds:
            check_leakage(subjects[tr], subjects[te])
            out = fit_predict(tr, te)
            pred, score = out[0], out[1]
            dec = out[2] if len(out) > 2 else None
            y_true_all.append(y[te])
            y_pred_all.append(pred)
            if dec is not None:
                dec_all.append(dec)
            if score is not None:
                score_all.append(score)
            per_fold.append({"fold": f"subject_{s}",
                             "test_subjects": [s],
                             **compute_metrics(y[te], pred,
                                               score if score is not None else None)})
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        sc = np.concatenate(score_all) if score_all else None
        results["models"][model_name] = {
            "fold_metrics": per_fold,
            "overall": compute_metrics(y_true, y_pred, sc),
        }
        if dec_all:
            dec = np.concatenate(dec_all)
            cov = dec != "uncertain"
            if cov.sum() > 4 and len(np.unique(y_true[cov])) == 2:
                cov_pred = np.where(dec[cov] == "high", 1, 0)
                results["models"][model_name]["coverage"] = float(cov.mean())
                results["models"][model_name]["covered_only"] = compute_metrics(
                    y_true[cov], cov_pred
                )
            else:
                results["models"][model_name]["coverage"] = float(cov.mean())

    # fuzzy variants (binary stress-vs-rest only; the fuzzy engine is a binary
    # scorer by design, so in the 3class task only the ML baselines run)
    if task == "binary":
        for name, spec in fuzzy_specs.items():
            def fp(tr, te, spec=spec):
                F = fuzzy_frame(df)
                if spec["personalize"]:
                    F = personalize(F)
                model = FuzzyStressClassifier(
                    rules_path=spec["rules"],
                    use_quality=spec["use_quality"],
                    threshold_mode="train_fit",
                )
                model.fit(F.iloc[tr], y[tr])
                pred = model.predict(F.iloc[te])
                score = model.score_continuous(F.iloc[te])
                return pred, score, model.decision_labels(F.iloc[te])
            run(name, fp)

    # baselines
    baselines = make_baselines(seed)
    for bname, bpipe in baselines.items():
        def fp(tr, te, bpipe=bpipe):
            m = clone(bpipe).fit(X.iloc[tr], y[tr])
            pred = m.predict(X.iloc[te])
            try:
                score = m.predict_proba(X.iloc[te]).max(axis=1)
            except Exception:
                score = None
            return pred, score
        run(bname, fp)

    # unimodal view baselines (wrist-only / chest-only SVM)
    Xc = df[CHEST_FEATURES].copy()
    for vname, Xv in (("svm_wrist_only", X), ("svm_chest_only", Xc)):
        def fp(tr, te, Xv=Xv):
            m = clone(make_baselines(seed)["svm_rbf"]).fit(Xv.iloc[tr], y[tr])
            return m.predict(Xv.iloc[te]), m.predict_proba(Xv.iloc[te])[:, 1]
        run(vname, fp)

    # fuzzy late fusion of calibrated unimodal evidence (leakage-safe:
    # fusion inputs for TRAINING subjects are inner-GroupKFold out-of-fold
    # scores; the test subject is scored by models that never saw it)
    import stress_fuzzy as _sf

    FUSION_RULES = Path(_sf.__file__).parent / "rules" / "fusion_rules.yaml"
    wrist_pipe = make_baselines(seed)["svm_rbf"]
    chest_pipe = make_baselines(seed)["svm_rbf"]

    def fusion_frame(w: np.ndarray, c: np.ndarray) -> pd.DataFrame:
        f = pd.DataFrame({"wrist_score": w, "chest_score": c})
        f["agreement"] = 1.0 - np.abs(f.wrist_score - f.chest_score)
        return f

    from sklearn.model_selection import GroupKFold as _GKF

    # fusion is a binary (stress vs rest) architecture by design; in the
    # 3class task only the multiclass baselines and unimodal views run
    if task == "binary":
        def class_conditional_terms(scores: np.ndarray, labels: np.ndarray) -> dict:
            """Score-space membership terms anchored on class-conditional
            quantiles of TRAINING out-of-fold scores: 'low' saturates where true
            negatives live, 'high' where true positives live. This is the
            statistically meaningful way to fuzzify a discriminant score; pooled
            quantiles of a ~30%-prevalence class put the 'high' term inside the
            baseline range and destroy specificity."""
            neg = scores[labels == 0]
            pos = scores[labels == 1]
            lo, hi = float(np.min(scores)), float(np.max(scores))
            neg_q90, neg_q99 = (float(np.quantile(neg, q)) for q in (0.90, 0.995))
            pos_q005, pos_q50 = (float(np.quantile(pos, q)) for q in (0.005, 0.50))
            if not neg_q90 < pos_q50:  # degenerate separation -> fall back wide
                mid = (lo + hi) / 2
                neg_q90, pos_q50 = lo, hi
                neg_q99, pos_q005 = mid, mid
            mid1, mid2 = (neg_q90 + pos_q005) / 2, (neg_q99 + pos_q50) / 2
            return {
                "low": {"type": "trapezoid",
                        "params": [lo, lo, neg_q90, max(neg_q99, mid1)]},
                "medium": {"type": "triangle",
                           "params": [neg_q90, (mid1 + mid2) / 2, pos_q50]},
                "high": {"type": "trapezoid",
                         "params": [min(pos_q005, mid2), pos_q50, hi, hi]},
            }

        for fname, use_quality in (("fuzzy_multimodal_fusion", True),
                                   ("fuzzy_fusion_no_conflict", False)):
            def fp(tr, te, use_quality=use_quality):
                oof_w = np.zeros(len(tr))
                oof_c = np.zeros(len(tr))
                gk = _GKF(n_splits=5)
                for itr, ite in gk.split(np.zeros(len(tr)), groups=subjects[tr]):
                    mw = clone(wrist_pipe).fit(X.iloc[tr[itr]], y[tr[itr]])
                    mc = clone(chest_pipe).fit(Xc.iloc[tr[itr]], y[tr[itr]])
                    oof_w[ite] = mw.predict_proba(X.iloc[tr[ite]])[:, 1]
                    oof_c[ite] = mc.predict_proba(Xc.iloc[tr[ite]])[:, 1]

                spec = load_rules(FUSION_RULES)
                rules = parse_rules(spec["rules"])
                if not use_quality:
                    rules = [r for r in rules if r.consequent != "uncertain"]
                terms = {
                    "wrist_score": class_conditional_terms(oof_w, y[tr]),
                    "chest_score": class_conditional_terms(oof_c, y[tr]),
                    "agreement": spec["fixed_inputs"]["agreement"]["terms"],
                }
                engine = FuzzyEngine(terms, spec["output_terms"], rules)

                # decision thresholds fitted on OOF training scores only
                oof_scores = np.array([
                    engine.infer(dict(row._asdict())).score or 0.5
                    for row in fusion_frame(oof_w, oof_c).itertuples(index=False)
                ])
                best, best_f1 = (0.35, 0.65), -1.0
                for cut_low in np.arange(0.30, 0.61, 0.05):
                    for cut_high in np.arange(max(cut_low + 0.10, 0.40), 0.71, 0.05):
                        mid = (oof_scores >= cut_low) & (oof_scores <= cut_high)
                        pred = np.where(oof_scores < cut_low, 0,
                                        np.where(oof_scores > cut_high, 1, -1))
                        pred[mid] = np.where(
                            oof_scores[mid] - cut_low <= cut_high - oof_scores[mid],
                            0, 1)
                        f1 = f1_score(y[tr], pred, average="macro", zero_division=0)
                        if f1 > best_f1:
                            best_f1, best = f1, (float(cut_low), float(cut_high))

                mw = clone(wrist_pipe).fit(X.iloc[tr], y[tr])
                mc = clone(chest_pipe).fit(Xc.iloc[tr], y[tr])
                F_te = fusion_frame(
                    mw.predict_proba(X.iloc[te])[:, 1],
                    mc.predict_proba(Xc.iloc[te])[:, 1],
                )
                outs = [engine.infer(dict(r._asdict())) for r in F_te.itertuples(index=False)]
                scores = np.array([o.score if o.score is not None else 0.5
                                   for o in outs])
                forced = (scores >= 0.5).astype(int)
                dec = np.where(scores <= best[0], "low",
                               np.where(scores >= best[1], "high", "uncertain"))
                # an explicit uncertain rule outweighing stress evidence -> uncertain
                for i, o in enumerate(outs):
                    if o.uncertain_mass >= o.stress_mass and o.uncertain_mass > 0:
                        dec[i] = "uncertain"
                return forced, scores, dec
            run(fname, fp)

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/WESAD")
    ap.add_argument("--out-dir", default="bench/wesad")
    ap.add_argument("--task", default="binary", choices=VALID_TASKS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data_dir)
    subj_dirs = sorted(p for p in root.iterdir()
                       if p.is_dir() and p.name.startswith("S") and p.name[1:].isdigit())
    if not subj_dirs:
        print(f"[error] no subject dirs (S2..S17) under {root}", file=sys.stderr)
        return 1
    print(f"[1/3] loading {len(subj_dirs)} subjects ({args.task} task) ...")
    frames = []
    for d in subj_dirs:
        try:
            frames.append(subject_windows(d, args.task))
        except Exception as exc:
            print(f"  [warn] skipping {d.name}: {exc}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df.label.notna()]
    print(f"[2/3] {len(df)} windows over {df.subject.nunique()} subjects "
          f"(classes: {df.label.value_counts().sort_index().to_dict()})")
    if df.subject.nunique() < 10:
        print("[error] too few subjects for a meaningful LOSO benchmark", file=sys.stderr)
        return 1

    print(f"[3/3] running LOSO ({args.task}) ...")
    results = loso_benchmark(df, args.task, args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"wesad_{args.task}_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(f"\n=== WESAD LOSO ({args.task}), {results['n_subjects']} subjects, "
          f"{results['n_windows']} windows ===")
    for name, m in results["models"].items():
        o = m["overall"]
        extra = (f" stress-recall {o['per_class']['stress']['recall']:.3f}"
                 if "stress" in o.get("per_class", {}) else "")
        print(f"  {name:24s} macro-F1 {o['macro_f1']:.3f}  "
              f"balanced-acc {o['balanced_accuracy']:.3f}{extra}")
    print(f"\n[ok] results: {out}/wesad_{args.task}_results.json")
    print("Exploratory research output - not a diagnostic tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
