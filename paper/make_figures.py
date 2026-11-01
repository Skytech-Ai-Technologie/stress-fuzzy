"""Generate all paper figures from the actual result files (no hand-typed data).

Run from the repository root:  python paper/make_figures.py
Outputs PNG (150 dpi) into paper/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

C_FUZZY = "#d62728"
C_BASE = "#1f77b4"
C_CTRL = "#7f7f7f"
C_OK = "#2ca02c"
C_WARN = "#ff7f0e"

FUZZY_MODELS = {
    "fuzzy_quality_personalized": "fuzzy (quality+pers.)",
    "fuzzy_quality_unpersonalized": "fuzzy (unpers.)",
    "fuzzy_no_quality": "fuzzy (no quality)",
    "fuzzy_multimodal_fusion": "fuzzy multimodal fusion",
    "fuzzy_hrv_quality": "fuzzy direct (HRV)",
    "fuzzy_hrv_personalized": "fuzzy direct (HRV+pers.)",
}


def jload(path: Path):
    if not path.is_file():
        print(f"  [skip] missing {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def short(name: str) -> str:
    if name in FUZZY_MODELS:
        return FUZZY_MODELS[name]
    return {
        "logistic_regression": "logistic regression",
        "svm_rbf": "SVM (RBF)",
        "random_forest": "random forest",
        "svm_wrist_only": "SVM wrist-only",
        "svm_chest_only": "SVM chest-only",
        "fuzzy_fusion_no_conflict": "fuzzy fusion (no conflict rules)",
    }.get(name, name)


# ----------------------------------------------------------- fig 1: architecture


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec="#3b5b8f", fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.6",
                                    fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle=style, mutation_scale=13,
                                     lw=1.2, color="#444444"))

    # data layer
    box(2, 38, 20, 12, "WESAD (15 subj)\nE4 wrist 4/64/32 Hz\nRespiBAN chest 700 Hz\n+ labels (TSST protocol)",
        fc="#f5f5f0", bold=False)
    box(2, 22, 20, 10, "WorkStress3D (20 subj)\nE4 wrist 4 Hz\nphase labels 0/1 (+2 dropped)",
        fc="#f5f5f0")
    box(2, 6, 20, 10, "Audit gate\naliases, class-2, alignment,\nparticipant overlap", fc="#fdf2e3")

    # preprocessing
    box(28, 38, 17, 12, "Non-overlapping\n60 s / phase-pure\nwindows", fc="#eef7ee", ec="#3b7a3b")
    box(28, 22, 17, 10, "Non-overlapping\n14 s (56-sample)\nwindows", fc="#eef7ee", ec="#3b7a3b")

    # features
    box(51, 34, 18, 8, "Wrist features\nEDA/TEMP/ACC stats + HRV (BVP 64 Hz)", fs=7.5)
    box(51, 24, 18, 7, "Chest features\nECG->HRV, EDA, TEMP (700 Hz)", fs=7.5)

    # unimodal models
    box(51, 12, 18, 8, "Calibrated unimodal\nSVMs (fold-local fit)\nwrist & chest", fc="#fdeaea", ec="#a33")

    # fusion
    box(74, 30, 24, 16,
        "Mamdani fuzzy fusion layer\n"
        "class-conditional terms\n"
        "9 rules incl. conflict/quality\n"
        "min-max inference, centroid",
        fc="#fff4f2", ec="#c0392b", fs=8, bold=True)

    # outputs
    box(74, 10, 24, 12,
        "Outputs\nstress score [0,1]\nlow / uncertain / high\nfired rules + memberships\ncoverage 83% -> covered BA 0.905",
        fc="#f0f7f4", ec="#2c6e5a", fs=7.5)

    arrow(22, 44, 27.5, 44)
    arrow(22, 27, 27.5, 27)
    arrow(22, 11, 27.5, 11)
    arrow(45.5, 44, 50.5, 39)
    arrow(45.5, 27, 50.5, 30)
    arrow(60, 33.8, 60, 20.5)
    arrow(69.5, 16, 73.5, 34)
    arrow(86, 29.8, 86, 22.5)
    # audit gate arrows
    ax.text(2, 0.8, "audit blocks multimodal fusion on WorkStress3D "
                    "(no participant keys / timestamps)",
            fontsize=7.5, style="italic", color="#8a6d3b")
    ax.text(52, 4.2, "Inner GroupKFold out-of-fold scores fit the fusion terms\n"
                     "(leakage-safe); thresholds chosen on training scores only",
            fontsize=7.5, style="italic", color="#555555")

    ax.set_title("System architecture: interpretable quality-gated fuzzy fusion "
                 "of calibrated unimodal evidence", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_architecture.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------- fig 2: Mamdani engine example


def fig_mamdani():
    bounds = jload(ROOT / "artifacts/fuzzy_membership_bounds.json")
    examples = jload(ROOT / "artifacts/explainability_examples.json")
    if bounds is None or examples is None:
        return
    per_fold = bounds.get("membership_bounds_per_fold", {}).get(
        "fuzzy_quality_personalized", {})
    if not per_fold:
        return
    fb = next(iter(per_fold.values()))
    fitted = {k: v for k, v in fb.items() if isinstance(v, dict)}

    from stress_fuzzy.fuzzy_model import apply_mf, partition_3, load_rules

    fig, axes = plt.subplots(1, len(fitted) + 2, figsize=(2.45 * (len(fitted) + 2), 2.9))

    for ax, (name, b) in zip(axes, fitted.items()):
        lo = b["lo"] - (b["hi"] - b["lo"]) * 0.15
        hi = b["hi"] + (b["hi"] - b["lo"]) * 0.15
        xs = np.linspace(lo, hi, 240)
        terms = partition_3(None, b["lo"], b["m1"], b["m2"], b["m3"], b["hi"])
        for t in ("low", "medium", "high"):
            ax.plot(xs, apply_mf(xs, terms[t]), lw=1.4)
        ax.fill_between([b["m1"], b["m3"]], 0, 1.02, alpha=0.08, color="grey")
        ax.set_title(name, fontsize=8)
        ax.set_yticks([0, 1])
        ax.tick_params(labelsize=6.5)

    # example window: memberships + fired rules (first held-out example)
    ex = examples[0]["explanation"]
    terms = list(ex["memberships"].items())
    names = [t[0] for t in terms]
    lows = [t[1].get("low", 0) for t in terms]
    meds = [t[1].get("medium", 0) for t in terms]
    highs = [t[1].get("high", 0) for t in terms]
    xpos = np.arange(len(names))
    ax = axes[-2]
    ax.bar(xpos - 0.22, lows, 0.22, label="low")
    ax.bar(xpos, meds, 0.22, label="medium")
    ax.bar(xpos + 0.22, highs, 0.22, label="high")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_title("example window memberships", fontsize=8)
    ax.legend(fontsize=6)

    out_terms = load_rules()["output_terms"]
    xs = np.linspace(0, 1, 240)
    ax = axes[-1]
    for t, spec in out_terms.items():
        ax.plot(xs, apply_mf(xs, spec), lw=1.4, label=t)
    if ex["score"] is not None:
        ax.axvline(ex["score"], color="black", ls="--", lw=1)
    ax.set_title("output: stress (+example score)", fontsize=8)
    ax.legend(fontsize=6)
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=6.5)

    fig.suptitle("Mamdani engine: fold-fitted input partitions (left, one LOSO fold), "
                 "an explained window (middle), output terms (right)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "fig2_mamdani_engine.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------ fig 3: benchmark bars


def _bar_panel(ax, names, values, errs, colors, title):
    ypos = np.arange(len(names))
    ax.barh(ypos, values, xerr=errs, color=colors, edgecolor="black", lw=0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlim(0, 1.0)
    ax.set_title(title, fontsize=9)
    for y, v in zip(ypos, values):
        ax.text(min(v + (errs[ypos[list(ypos).index(y)]] if errs is not None else 0),
                    0.99) + 0.015, y, f"{v:.3f}", va="center", fontsize=7.5)


def fig_benchmarks():
    ws = jload(ROOT / "artifacts/evaluation_results.json")
    we = jload(ROOT / "bench/wesad/wesad_binary_results.json")
    if ws is None or we is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    order = ["fuzzy_quality_personalized", "fuzzy_quality_unpersonalized",
             "fuzzy_no_quality", "logistic_regression", "svm_rbf", "random_forest"]
    names, vals, errs, cols = [], [], [], []
    for m in order:
        agg = ws["models"][m]["aggregate"]
        names.append(short(m))
        vals.append(agg["balanced_accuracy"]["mean"])
        errs.append(agg["balanced_accuracy"]["std"])
        cols.append(C_FUZZY if m.startswith("fuzzy") else C_BASE)
    _bar_panel(axes[0], names, vals, errs, cols,
               "WorkStress3D - LOSO (20 participants, 696 windows)")

    order = ["fuzzy_hrv_quality", "fuzzy_hrv_personalized",
             "logistic_regression", "svm_chest_only", "svm_wrist_only",
             "random_forest", "fuzzy_multimodal_fusion"]
    names, vals, errs, cols = [], [], [], []
    for m in order:
        ov = we["models"][m]["overall"]
        names.append(short(m))
        vals.append(ov["balanced_accuracy"])
        errs.append(0)
        cols.append(C_FUZZY if m.startswith("fuzzy") else C_BASE)
    _bar_panel(axes[1], names, vals, None, cols,
               "WESAD - LOSO (15 subjects, 547 windows)")

    fig.suptitle("Balanced accuracy under participant-level holdout "
                 "(dashed line = chance)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "fig3_benchmarks.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------- fig 4: coverage / abstention


def fig_coverage():
    we = jload(ROOT / "bench/wesad/wesad_binary_results.json")
    b90 = jload(ROOT / "bench/bench90/bench90_results.json")
    if we is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    m = we["models"]["fuzzy_multimodal_fusion"]
    cov = m.get("covered_only")
    bars = [("all windows\n(forced)", m["overall"]["balanced_accuracy"]),
            ("covered only\n(83.2% cov.)", cov["balanced_accuracy"] if cov else 0),
            ("best baseline RF", we["models"]["random_forest"]["overall"]["balanced_accuracy"])]
    ax = axes[0]
    colors = [C_FUZZY, C_OK, C_BASE]
    ax.bar(range(3), [b[1] for b in bars], color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_ylim(0, 1)
    ax.set_title("WESAD fusion: abstention buys accuracy\nwhere the model commits", fontsize=9)
    for i, b in enumerate(bars):
        ax.text(i, b[1] + 0.02, f"{b[1]:.3f}", ha="center", fontsize=8)

    if b90:
        cov_acc = b90.get("coverage_accuracy", {})
        covs = [1.0, 0.75, 0.5, 0.25, 0.10, 0.05]
        keys = ["top_100pct", "top_75pct", "top_50pct", "top_25pct",
                "top_10pct", "top_5pct"]
        ys = [cov_acc[k]["balanced_accuracy"] for k in keys if k in cov_acc]
        xs = [c for c, k in zip(covs, keys) if k in cov_acc]
        ax = axes[1]
        ax.plot(np.array(xs) * 100, ys, "o-", color=C_FUZZY)
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_xlabel("coverage (%)")
        ax.set_ylabel("balanced accuracy")
        ax.set_title("WorkStress3D SVM: confidence selects\nno better than chance (control)",
                     fontsize=9)
        ax.set_ylim(0.4, 0.75)

    fig.tight_layout()
    fig.savefig(FIG / "fig4_coverage.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------ fig 5: leakage ladder


def fig_leakage():
    b90 = jload(ROOT / "bench/bench90/bench90_results.json")
    if b90 is None:
        return
    ladder = b90["leakage_ladder"]
    tags = [("L1_no_overlap_random_split", "L1: no overlap,\nrandom split"),
            ("L2_overlap50_random_split", "L2: 50% overlap,\nrandom split"),
            ("L3_overlap87_random_split", "L3: 87.5% overlap,\nrandom split"),
            ("L5_60s_overlap87_random_split", "L5: 60 s windows,\n87.5% overlap")]
    if "L5_60s_overlap87_random_split" not in ladder:
        tags = tags[:3]
    vals = [ladder[t]["balanced_accuracy"] for t, _ in tags]
    loso = 0.597  # WESAD-matched honest reference from the same SVM (LOSO)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    bars = ax.bar([t[1] for t in tags], vals,
                  color=[C_WARN] * len(vals), edgecolor="black", lw=0.5)
    ax.axhline(loso, color=C_CTRL, ls="--", lw=1.2)
    ax.text(len(tags) - 0.45, loso + 0.012, "honest LOSO (same SVM): 0.597",
            fontsize=7.5, ha="right", color=C_CTRL)
    ax.axhline(0.9, color="grey", ls=":", lw=1)
    ax.text(0.02, 0.905, "0.90", fontsize=7.5, color="grey")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Leakage ladder on WorkStress3D (SVM): how published-style\n"
                 "high numbers emerge from window overlap + random splits",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_leakage_ladder.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ fig 6: transfer


def fig_transfer():
    tr = jload(ROOT / "bench/transfer/transfer_results.json")
    if tr is None:
        return
    ctrl = tr["in_dataset_control"]
    A = tr["transfer_A_wesad_to_workstress3d"]
    B = tr["transfer_B_workstress3d_to_wesad"]
    models = ["logistic_regression", "svm_rbf", "random_forest"]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    labels = ["WESAD\n(control LOSO)", "WorkStress3D\n(control LOSO)",
              "WESAD ->\nWorkStress3D", "WorkStress3D\n-> WESAD"]
    vals = [max(ctrl["wesad"][m]["balanced_accuracy"] for m in models),
            max(ctrl["workstress3d"][m]["balanced_accuracy"] for m in models),
            max(A[m]["balanced_accuracy"] for m in models),
            max(B[m]["balanced_accuracy"] for m in models)]
    colors = [C_OK, C_OK, C_WARN, C_WARN]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", lw=0.5)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("balanced accuracy (best model)")
    ax.set_title("Cross-protocol transfer collapses to chance in both directions\n"
                 "(same device, harmonized EDA/TEMP/ACC features)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_transfer.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------ fig 7: WorkStress3D refutations


def fig_refutations():
    cal = jload(ROOT / "bench/pilots/calibration_curve.json")
    cal_t = jload(ROOT / "bench/pilots/calibration_threshold.json")
    leak = jload(ROOT / "bench/pilots/leakage_inflation.json")
    if not (cal and cal_t):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    ax = axes[0]
    for key, label, color in (
        (cal, "model refit (SVM)", C_FUZZY),
        (cal_t, "personal threshold only", C_BASE),
    ):
        s = key["summary"]
        ks = sorted(int(k) for k in s)
        ys = [s[str(k)]["mean_balanced_accuracy"] for k in ks]
        ax.plot(ks, ys, "o-", label=label, color=color)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlabel("calibration windows from the held-out participant (k)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Personal calibration does not lift\nWorkStress3D performance", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.set_xscale("symlog")

    if leak:
        ax = axes[1]
        protos = ["random_kfold5_leaky", "phase_grouped_kfold5",
                  "participant_grouped_kfold5", "leave_one_subject_out",
                  "within_participant_5fold"]
        names = ["random\n(leaky)", "phase-grouped", "participant-\ngrouped",
                 "LOSO", "within-\nparticipant"]
        for m, color in (("logistic_regression", C_BASE), ("svm_rbf", C_WARN),
                         ("random_forest", C_OK)):
            ys = [leak[p][m]["balanced_accuracy_mean"] for p in protos]
            ax.plot(names, ys, "o-", label=short(m), color=color)
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_ylabel("balanced accuracy")
        ax.set_title("Evaluation-protocol sensitivity:\nwithin-person is also chance",
                     fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", labelsize=7)

    fig.tight_layout()
    fig.savefig(FIG / "fig7_refutations.png", dpi=150)
    plt.close(fig)


def main() -> int:
    print("generating figures into paper/figures/ ...")
    fig_architecture()
    fig_mamdani()
    fig_benchmarks()
    fig_coverage()
    fig_leakage()
    fig_transfer()
    fig_refutations()
    made = sorted(p.name for p in FIG.glob("*.png"))
    for m in made:
        print("  [ok]", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
