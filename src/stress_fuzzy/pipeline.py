"""End-to-end pipeline: audit -> windowing -> features -> grouped evaluation
-> reports.

CLI:
    python -m stress_fuzzy.pipeline --data-dir database [--protocol loso|gk5]
        [--window-file 15sn|30sn|60sn] [--no-personalize] [--no-quality]
        [--out-dir .] [--rules path.yaml]

Multimodal fusion stays blocked until alignment is demonstrated; passing
--allow-multimodal without alignment evidence refuses with a diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import (
    DataError,
    check_alignment,
    load_physiological,
    segment_windows,
)
from .evaluation import run_experiment
from .features import extract_features
from .fuzzy_model import FuzzyStressClassifier, build_fuzzy_inputs
from .reporting import (
    write_environment,
    write_explanations,
    write_markdown_report,
    write_membership_figure,
    write_metrics_tables,
)
from .validation import run_audit, write_audit_outputs


def collect_explanations(features: pd.DataFrame, cfg: C.Config,
                         n_subjects: int = 3) -> list[dict]:
    """Explain genuinely held-out windows: refit the per-fold model of the
    first LOSO folds and explain selected test windows of the held-out
    participant (never seen during fitting)."""
    examples: list[dict] = []
    subjects = sorted(features.subject.unique())
    fuzzy_X = build_fuzzy_inputs(features, personalize=cfg.personalize)
    for subj in subjects[:n_subjects]:
        te_mask = features.subject == subj
        tr_mask = ~te_mask
        model = FuzzyStressClassifier(
            rules_path=cfg.rules_path or None,
            use_quality=cfg.quality_aware,
            threshold_mode=cfg.threshold_mode,
            fixed_thresholds=cfg.fixed_thresholds,
            confidence_tau=cfg.confidence_tau,
        )
        model.fit(fuzzy_X[tr_mask], features.label[tr_mask])
        cand = features[te_mask].copy()
        if cand.empty:
            continue
        picks = []
        for lab in (0, 1):
            sub = cand[cand.label == lab]
            if len(sub):
                picks.append(sub.quality.idxmax())
        if len(cand):
            picks.append(cand.quality.idxmin())
        for idx in dict.fromkeys(picks):  # unique, keep order
            row = fuzzy_X.loc[idx]
            examples.append(
                {
                    "subject": int(features.loc[idx, "subject"]),
                    "fold": f"subject_{features.loc[idx, 'subject']}",
                    "true_label": int(features.loc[idx, "label"]),
                    "window_quality": float(features.loc[idx, "quality"]),
                    "held_out": True,
                    "explanation": model.explain(row.to_dict()),
                }
            )
    return examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stress_fuzzy.pipeline",
        description="Run the full grouped evaluation and export reports.",
    )
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--window-file", default=C.DEFAULT_WINDOW_FILE,
                        choices=list(C.PHYSIO_FILES))
    parser.add_argument("--window-samples", type=int, default=0,
                        help="window length in samples (4 Hz); default: the "
                        "file's natural base unit (GCD of its phase lengths, "
                        "56/118/240 for the 15/30/60 s files)")
    parser.add_argument("--protocol", default="loso", choices=["loso", "gk5"])
    parser.add_argument("--no-personalize", action="store_true")
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--rules", default="")
    parser.add_argument("--allow-multimodal", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args(argv)

    cfg = C.Config(
        window_file=args.window_file,
        protocol=args.protocol,
        personalize=not args.no_personalize,
        quality_aware=not args.no_quality,
        rules_path=args.rules,
        allow_multimodal=args.allow_multimodal,
    )
    data_dir, notes = C.resolve_data_dir(args.data_dir)
    for n in notes:
        print(f"[info] {n}")
    cfg.data_dir = str(data_dir)
    reports_dir = Path(args.out_dir) / "reports"
    artifacts_dir = Path(args.out_dir) / "artifacts"

    # ---- 1. audit gate ------------------------------------------------
    if not args.skip_audit:
        print("[1/5] running data audit ...")
        audit = run_audit(data_dir)
        write_audit_outputs(audit, reports_dir, artifacts_dir)
        if not audit["ok_for_modelling"]:
            print(
                "[blocked] audit says the data is not ready for modelling "
                "(see reports/data_audit.md); fix the blocking findings first.",
                file=sys.stderr,
            )
            return 1
    else:
        print("[1/5] audit skipped (--skip-audit)")

    # ---- 2. multimodal gate -------------------------------------------
    alignment = check_alignment(data_dir)
    if cfg.allow_multimodal and not alignment.multimodal_allowed:
        print(
            "[blocked] --allow-multimodal given, but alignment cannot be "
            "demonstrated:\n  - " + "\n  - ".join(alignment.blockers) +
            "\nPhysiological-only evaluation continues to be the default; "
            "multimodal training is refused.",
            file=sys.stderr,
        )
        return 1
    if alignment.blockers:
        print("[info] multimodal fusion blocked (no alignment evidence):")
        for b in alignment.blockers:
            print(f"       - {b}")

    # ---- 3. load + window + features ----------------------------------
    print(f"[2/5] loading physiological file {cfg.window_file} ...")
    try:
        raw = load_physiological(data_dir, cfg.window_file)
    except DataError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    # Derive the natural base window (GCD of contiguous phase lengths) unless
    # the user pinned --window-samples: this makes the 15/30/60 s files true
    # window-length variants instead of the same 14 s window on more rows.
    import numpy as np

    from .data import phase_stats

    natural = int(np.gcd.reduce(phase_stats(raw).length.astype("int64").values))
    if args.window_samples > 0:
        cfg.window_samples = args.window_samples
    else:
        cfg.window_samples = natural
    print(f"[3/5] segmenting windows (window_samples={cfg.window_samples}, "
          f"natural base unit of this file: {natural}) and extracting features ...")
    widx = segment_windows(raw, cfg.window_samples, tuple(cfg.drop_labels))
    features = extract_features(widx)
    s = widx.summarize()
    print(
        f"       {s['n_windows']} windows over {s['n_participants']} "
        f"participants (labels: {s['label_counts']}; "
        f"dropped ambiguous rows: {s['dropped_label_rows']}, "
        f"phase-tail rows: {s['dropped_tail_rows']})"
    )
    if s["n_participants"] < 6:
        print(
            "[blocked] fewer than 6 participants with data - grouped "
            "evaluation would be meaningless",
            file=sys.stderr,
        )
        return 1

    # ---- 4. grouped evaluation ----------------------------------------
    print(f"[4/5] running grouped evaluation (protocol={cfg.protocol}) ...")
    results = run_experiment(features, cfg)
    examples = collect_explanations(features, cfg)

    # ---- 5. exports ----------------------------------------------------
    print("[5/5] writing artifacts and reports ...")
    write_environment(cfg, artifacts_dir)
    write_metrics_tables(results, artifacts_dir)
    write_explanations(examples, artifacts_dir)
    report = write_markdown_report(results, cfg, reports_dir, examples)
    write_membership_figure(results, reports_dir)
    from .reporting import write_confusion_figures

    write_confusion_figures(results, reports_dir)

    agg = results["models"]["fuzzy_quality_personalized"]["aggregate"]
    print()
    print("=== summary (exploratory, grouped protocol "
          f"{results['protocol']}, {results['n_participants']} participants, "
          f"{results['n_windows']} windows) ===")
    for name, mres in results["models"].items():
        a = mres["aggregate"]
        print(
            f"  {name:32s} macro-F1 {a['macro_f1']['mean']:.3f}±{a['macro_f1']['std']:.3f}"
            f"  balanced-acc {a['balanced_accuracy']['mean']:.3f}±{a['balanced_accuracy']['std']:.3f}"
            f"  stress-recall {a['stress_recall']['mean']:.3f}"
            + (f"  coverage {a['coverage']['mean']:.2f}" if mres["kind"] == "fuzzy" else "")
        )
    print()
    print(
        "Research prototype only - not a medical or employee-monitoring "
        "tool. With 20 participants these numbers are exploratory."
    )
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
