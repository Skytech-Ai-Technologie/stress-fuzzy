"""Machine- and human-readable output writing.

Artifacts (JSON/CSV, machine-readable):
    evaluation_results.json, fold_metrics.csv, participant_metrics.csv,
    fuzzy_membership_bounds.json, explainability_examples.json,
    environment.json
Reports (Markdown + PNG, human-readable):
    evaluation_report.md, confusion_<model>.png, membership_functions.png
"""
from __future__ import annotations

import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from . import config as C


def jsonable(obj):
    """Recursively convert NaN/inf floats to None so outputs are strict JSON."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return obj


def dump_json(obj, path: Path) -> None:
    path.write_text(json.dumps(jsonable(obj), indent=2, default=str),
                    encoding="utf-8")


def _versions() -> dict:
    import sklearn
    import scipy
    import yaml

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "stress_fuzzy": __version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "pyyaml": yaml.__version__,
    }


def write_environment(cfg: C.Config, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "config": cfg.to_dict(),
        "rules_file": cfg.rules_path or "bundled: stress_fuzzy/rules/fuzzy_rules.yaml",
    }
    dump_json(payload, artifacts_dir / "environment.json")


def write_metrics_tables(results: dict, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, mres in results["models"].items():
        for fr in mres["fold_metrics"]:
            rows.append(
                {
                    "model": model,
                    "fold": fr["fold"],
                    "test_subjects": ",".join(str(s) for s in fr["test_subjects"]),
                    "n": fr["n"],
                    "macro_f1": fr["macro_f1"],
                    "balanced_accuracy": fr["balanced_accuracy"],
                    "stress_recall": fr["stress_recall"],
                    "pr_auc": fr["pr_auc"],
                    "accuracy": fr["accuracy"],
                    "coverage": fr["coverage"],
                }
            )
    pd.DataFrame(rows).to_csv(artifacts_dir / "fold_metrics.csv", index=False)

    prows = []
    for model, mres in results["models"].items():
        for pr in mres["participant_metrics"]:
            prows.append({"model": model, **pr})
    pd.DataFrame(prows).to_csv(artifacts_dir / "participant_metrics.csv", index=False)

    dump_json(
        {
            "membership_bounds_per_fold": results["fold_fuzzy_bounds"],
            "thresholds_per_fold": results["fold_thresholds"],
            "note": "bounds estimated from training participants of each "
            "fold only (25/50/75 percentiles); 'fixed' inputs use the "
            "parameters declared in the rules YAML",
        },
        artifacts_dir / "fuzzy_membership_bounds.json",
    )

    dump_json(results, artifacts_dir / "evaluation_results.json")


def write_explanations(examples: list[dict], artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dump_json(examples, artifacts_dir / "explainability_examples.json")


def _fmt_agg(agg: dict | None, key: str) -> str:
    if not agg or agg.get(key) is None:
        return "n/a"
    a = agg[key]
    return f"{a['mean']:.3f} ± {a['std']:.3f}"


def write_markdown_report(
    results: dict,
    cfg: C.Config,
    reports_dir: Path,
    examples: list[dict],
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    L.append("# Stress detection evaluation report")
    L.append("")
    L.append(
        f"Generated: {datetime.now(timezone.utc).isoformat()}  |  "
        f"protocol: **{results['protocol']}**  |  "
        f"participants: **{results['n_participants']}**  |  "
        f"windows: **{results['n_windows']}**"
    )
    L.append("")
    L.append(
        "> Research prototype for a reproducible experiment on WorkStress3D. "
        "Not a medical device, not a diagnostic tool, and not usable for "
        "employment decisions. With 20 participants, results below are "
        "**exploratory**, not confirmatory."
    )
    L.append("")
    L.append(f"Class balance (windows): {results['class_counts']}")
    L.append("")

    L.append("## Model comparison (mean ± std over folds)")
    L.append("")
    L.append("| model | macro-F1 | balanced acc | stress recall | PR-AUC | coverage |")
    L.append("|---|---|---|---|---|---|")
    for name, mres in results["models"].items():
        agg = mres["aggregate"]
        L.append(
            f"| {name} | {_fmt_agg(agg,'macro_f1')} | {_fmt_agg(agg,'balanced_accuracy')} "
            f"| {_fmt_agg(agg,'stress_recall')} | {_fmt_agg(agg,'pr_auc')} "
            f"| {_fmt_agg(agg,'coverage')} |"
        )
    L.append("")
    L.append(
        "Coverage = fraction of windows where the model commits to a label; "
        "baselines always commit (1.0). The fuzzy model abstains via its "
        "'uncertain' decision when quality is poor, signals conflict, or "
        "confidence is low."
    )
    L.append("")

    L.append("## Coverage vs performance (fuzzy)")
    L.append("")
    fuzzy = results["models"].get("fuzzy_quality_personalized")
    if fuzzy:
        ov = fuzzy["overall_metrics"]
        cov = fuzzy.get("covered_only_metrics")
        L.append("| metric | all windows (forced) | decided windows only |")
        L.append("|---|---|---|")
        L.append(f"| n windows | {ov['n']} | {cov['n'] if cov else 0} |")
        if cov:
            L.append(f"| macro-F1 | {ov['macro_f1']:.3f} | {cov['macro_f1']:.3f} |")
            L.append(f"| balanced acc | {ov['balanced_accuracy']:.3f} | "
                     f"{cov['balanced_accuracy']:.3f} |")
        else:
            L.append(f"| macro-F1 | {ov['macro_f1']:.3f} | n/a |")
            L.append(f"| balanced acc | {ov['balanced_accuracy']:.3f} | n/a |")
        agg = fuzzy["aggregate"]
        L.append("")
        L.append(f"Mean coverage over folds: {_fmt_agg(agg,'coverage')}")
    L.append("")

    L.append("## Ablations")
    L.append("")
    L.append("| variant | macro-F1 | balanced acc | stress recall |")
    L.append("|---|---|---|---|")
    for name in ("fuzzy_quality_personalized", "fuzzy_quality_unpersonalized",
                 "fuzzy_no_quality"):
        mres = results["models"].get(name)
        if not mres:
            continue
        agg = mres["aggregate"]
        L.append(
            f"| {name} | {_fmt_agg(agg,'macro_f1')} | "
            f"{_fmt_agg(agg,'balanced_accuracy')} | {_fmt_agg(agg,'stress_recall')} |"
        )
    L.append("")

    L.append("## Confusion matrices (all held-out windows pooled)")
    L.append("")
    for name, mres in results["models"].items():
        cm = mres["overall_metrics"]["confusion_matrix"]
        L.append(f"**{name}** (rows = true non-stress/stress)")
        L.append("")
        L.append("| | pred non-stress | pred stress |")
        L.append("|---|---|---|")
        L.append(f"| true non-stress | {cm[0][0]} | {cm[0][1]} |")
        L.append(f"| true stress | {cm[1][0]} | {cm[1][1]} |")
        L.append("")

    L.append("## Participant-level results")
    L.append("")
    L.append(
        "Per-participant balanced accuracy is in "
        "`artifacts/participant_metrics.csv`; fold-level details in "
        "`artifacts/fold_metrics.csv`."
    )
    L.append("")
    L.append("## Statistical comparisons (exploratory)")
    L.append("")
    if results.get("comparisons"):
        L.append(
            "Wilcoxon signed-rank on per-participant balanced accuracy, "
            "fuzzy_quality_personalized vs each other model:"
        )
        L.append("")
        for name, c in results["comparisons"].items():
            if "p_value" in c:
                L.append(
                    f"- {name}: n={c['n_pairs']}, median diff "
                    f"{c['median_diff']:+.3f}, p={c['p_value']:.3f} "
                    f"({c['interpretation']})"
                )
            else:
                L.append(f"- {name}: {c}")
    L.append("")

    L.append("## Held-out prediction explanations (examples)")
    L.append("")
    for ex in examples[:6]:
        L.append(
            f"- Subject {ex['subject']} (true: {ex['true_label']}) -> "
            f"decision **{ex['explanation']['decision']}**, score "
            f"{None if ex['explanation']['score'] is None else round(ex['explanation']['score'],3)}, "
            f"confidence {ex['explanation']['confidence']:.2f}; fired rules: "
            + "; ".join(
                f"{r['id']}({r['consequent']}, {r['strength']:.2f})"
                for r in ex["explanation"]["fired_rules"][:3]
            )
        )
    L.append("")
    L.append("Full explanations: `artifacts/explainability_examples.json`.")
    L.append("")

    L.append("## Membership bounds")
    L.append("")
    L.append(
        "Fuzzy membership limits were estimated from training participants "
        "only, per fold: see `artifacts/fuzzy_membership_bounds.json` and "
        "`reports/membership_functions.png`."
    )
    L.append("")

    L.append("## Known limitations")
    L.append("")
    L.append("- 20 participants; every number here is exploratory.")
    L.append(
        "- Labels come from the dataset's phase-structured protocol "
        "(emotion 0/1); the undocumented value 2 was excluded (see audit)."
    )
    L.append(
        "- Audio and facial modalities are blocked: no participant keys, no "
        "timestamps, no alignment evidence (see reports/data_audit.md)."
    )
    L.append(
        "- Fuzzy membership values are NOT calibrated probabilities; the "
        "score is a rule-based index in [0,1]."
    )
    L.append(
        "- Results are not comparable to other papers' numbers on this or "
        "other datasets without identical splits and protocols."
    )
    L.append("")

    out = reports_dir / "evaluation_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def write_membership_figure(results: dict, reports_dir: Path) -> None:
    """Plot one representative fold's fitted input MFs + output terms."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    from .fuzzy_model import apply_mf, partition_3

    bounds = results.get("fold_fuzzy_bounds", {}).get("fuzzy_quality_personalized", {})
    if not bounds:
        return
    fold_name = next(iter(bounds))
    fb = bounds[fold_name]
    fitted = {k: v for k, v in fb.items() if isinstance(v, dict)}
    if not fitted:
        return
    fig, axes = plt.subplots(1, len(fitted) + 1, figsize=(3.2 * (len(fitted) + 1), 2.8))
    for ax, (name, b) in zip(axes[:-1], fitted.items()):
        lo = b["lo"] - (b["hi"] - b["lo"]) * 0.1
        hi = b["hi"] + (b["hi"] - b["lo"]) * 0.1
        xs = np.linspace(lo, hi, 200)
        terms = partition_3(None, b["lo"], b["m1"], b["m2"], b["m3"], b["hi"])
        for t in ("low", "medium", "high"):
            ax.plot(xs, apply_mf(xs, terms[t]), label=t)
        ax.set_title(f"{name}\n(fold {fold_name})")
        ax.set_yticks([0, 1])
        ax.legend(fontsize=7)
    xs = np.linspace(0, 1, 200)
    from .fuzzy_model import load_rules

    out_terms = load_rules()["output_terms"]
    for t, spec in out_terms.items():
        axes[-1].plot(xs, apply_mf(xs, spec), label=t)
    axes[-1].set_title("output: stress")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(reports_dir / "membership_functions.png", dpi=140)
    plt.close(fig)


def write_confusion_figures(results: dict, reports_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name, mres in results["models"].items():
        cm = np.array(mres["overall_metrics"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(3.2, 2.8))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="black")
        ax.set_xticks([0, 1], ["pred non-stress", "pred stress"])
        ax.set_yticks([0, 1], ["true non-stress", "true stress"])
        ax.set_title(name)
        fig.colorbar(im)
        fig.tight_layout()
        safe = name.replace("/", "_")
        fig.savefig(reports_dir / f"confusion_{safe}.png", dpi=140)
        plt.close(fig)
