"""Streamlit dashboard for the WorkStress3D fuzzy stress prototype.

Run:  streamlit run app.py

Reads the artifacts produced by:
    python -m stress_fuzzy.validation --data-dir <dir>
    python -m stress_fuzzy.pipeline    --data-dir <dir>

Research prototype only - not a diagnostic tool and not an employee
monitoring tool. No person-level sensitive information is displayed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="WorkStress3D fuzzy stress prototype", layout="wide")

st.title("Explainable fuzzy stress detection — WorkStress3D")
st.caption(
    "Research prototype. Not a medical device, not a diagnostic tool, and not "
    "usable for employment decisions. Participant identifiers are shown only "
    "as anonymous numbers for grouped evaluation."
)

ART = Path("artifacts")
REP = Path("reports")


@st.cache_data(show_spinner=False)
def load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: Path):
    if not path.is_file():
        return None
    return pd.read_csv(path)


audit = load_json(ART / "data_audit.json")
results = load_json(ART / "evaluation_results.json")
explanations = load_json(ART / "explainability_examples.json")
bounds = load_json(ART / "fuzzy_membership_bounds.json")
fold_metrics = load_csv(ART / "fold_metrics.csv")
participant_metrics = load_csv(ART / "participant_metrics.csv")

tabs = st.tabs(
    ["Data audit", "Evaluation", "Explain a prediction", "Export"]
)

# ------------------------------------------------------------------ audit
with tabs[0]:
    st.header("Data audit")
    if audit is None:
        st.info(
            "No audit found. Run `python -m stress_fuzzy.validation "
            "--data-dir <dir>` first."
        )
    else:
        verdict = (
            "OK for physiological modelling (multimodal blocked)"
            if audit["ok_for_modelling"]
            else "NOT ready for modelling"
        )
        if audit["ok_for_modelling"]:
            st.success(verdict)
        else:
            st.error(verdict)
        c1, c2, c3 = st.columns(3)
        n_phys = len(audit.get("physiological", {}))
        c1.metric("physiological files found", n_phys)
        if n_phys:
            p15 = audit["physiological"].get("15sn", {})
            c2.metric("participants", p15.get("n_participants", "n/a"))
            c3.metric("rows (15sn)", p15.get("rows", "n/a"))
        st.subheader("Files")
        st.dataframe(
            pd.DataFrame(audit["inventory"]), use_container_width=True, hide_index=True
        )
        st.subheader("Findings")
        for f in audit["findings"]:
            msg = f["message"]
            if f["severity"] == "blocking":
                st.error(msg)
            elif f["severity"] == "warning":
                st.warning(msg)
            else:
                st.info(msg)
        st.subheader("Label distribution (15sn file)")
        if n_phys and audit["physiological"]["15sn"].get("label_counts"):
            lc = audit["physiological"]["15sn"]["label_counts"]
            st.bar_chart(pd.DataFrame({"windows": lc}))
            st.write(
                "Assumed names: 0 = non-stress, 1 = stress, 2 = ambiguous "
                "(undocumented; excluded)"
            )
        st.subheader("Alignment status")
        al = audit.get("alignment", {})
        st.write(
            f"Multimodal allowed: **{al.get('multimodal_allowed', False)}**"
        )
        for b in al.get("blockers", []):
            st.warning(f"blocker: {b}")
        for n in al.get("notes", []):
            st.caption(n)
        st.subheader("Assumptions")
        for s in audit.get("assumptions", []):
            st.markdown(f"- {s}")

# ------------------------------------------------------------- evaluation
with tabs[1]:
    st.header("Evaluation (grouped, participant-level holdout)")
    if results is None:
        st.info(
            "No results found. Run `python -m stress_fuzzy.pipeline "
            "--data-dir <dir>` first."
        )
    else:
        st.write(
            f"Protocol **{results['protocol']}** | "
            f"participants **{results['n_participants']}** | "
            f"windows **{results['n_windows']}** | "
            f"class counts {results['class_counts']}"
        )
        st.caption(
            "With 20 participants these numbers are exploratory, not "
            "confirmatory."
        )
        rows = []
        for name, m in results["models"].items():
            a = m["aggregate"]
            rows.append(
                {
                    "model": name,
                    "macro-F1": f"{a['macro_f1']['mean']:.3f} ± {a['macro_f1']['std']:.3f}",
                    "balanced acc": f"{a['balanced_accuracy']['mean']:.3f} ± "
                                    f"{a['balanced_accuracy']['std']:.3f}",
                    "stress recall": f"{a['stress_recall']['mean']:.3f}",
                    "PR-AUC": ("n/a" if a["pr_auc"] is None else
                               f"{a['pr_auc']['mean']:.3f}"),
                    "coverage": ("n/a" if a["coverage"] is None else
                                 f"{a['coverage']['mean']:.2f}"),
                }
            )
        st.subheader("Model comparison (mean ± std over folds)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Coverage vs performance (fuzzy)")
            fuzzy = results["models"].get("fuzzy_quality_personalized")
            if fuzzy:
                ov = fuzzy["overall_metrics"]
                cov = fuzzy.get("covered_only_metrics")
                st.write(
                    f"All windows (forced): macro-F1 **{ov['macro_f1']:.3f}**, "
                    f"balanced acc **{ov['balanced_accuracy']:.3f}** (n={ov['n']})"
                )
                if cov:
                    st.write(
                        f"Decided windows only: macro-F1 **{cov['macro_f1']:.3f}**, "
                        f"balanced acc **{cov['balanced_accuracy']:.3f}** "
                        f"(n={cov['n']})"
                    )
                cm = ov["confusion_matrix"]
                st.table(
                    pd.DataFrame(
                        cm,
                        columns=["pred non-stress", "pred stress"],
                        index=["true non-stress", "true stress"],
                    )
                )
        with right:
            st.subheader("Comparisons (exploratory Wilcoxon)")
            comps = results.get("comparisons", {})
            if comps:
                crows = []
                for name, c in comps.items():
                    crows.append(
                        {
                            "vs": name,
                            "n pairs": c.get("n_pairs"),
                            "median diff": c.get("median_diff"),
                            "p": c.get("p_value", c.get("test")),
                        }
                    )
                st.dataframe(pd.DataFrame(crows), hide_index=True)
            else:
                st.write("n/a")

        if fold_metrics is not None:
            st.subheader("Fold-level metrics")
            model_filter = st.multiselect(
                "models", sorted(fold_metrics.model.unique()),
                default=["fuzzy_quality_personalized", "logistic_regression"],
            )
            st.dataframe(
                fold_metrics[fold_metrics.model.isin(model_filter)],
                use_container_width=True,
                hide_index=True,
            )
            st.line_chart(
                fold_metrics[fold_metrics.model.isin(model_filter)]
                .pivot(index="fold", columns="model", values="balanced_accuracy")
            )
        if participant_metrics is not None:
            st.subheader("Participant-level balanced accuracy")
            st.dataframe(
                participant_metrics.pivot(
                    index="subject", columns="model", values="balanced_accuracy"
                ),
                use_container_width=True,
            )
            st.caption(
                "Each row is one held-out participant (LOSO) or a participant "
                "held out within a GroupKFold split."
            )

# -------------------------------------------------------------- explain
with tabs[2]:
    st.header("Explain a held-out prediction")
    if not explanations:
        st.info(
            "No explanations found. Run the pipeline first; explanations are "
            "generated only for windows of participants held out from training."
        )
    else:
        subjects = sorted({e["subject"] for e in explanations})
        sel_subject = st.selectbox("participant (held out)", subjects)
        cases = [e for e in explanations if e["subject"] == sel_subject]
        case = st.selectbox(
            "window",
            range(len(cases)),
            format_func=lambda i: (
                f"true={'stress' if cases[i]['true_label'] else 'non-stress'}, "
                f"quality={cases[i]['window_quality']:.2f}"
            ),
        )
        ex = cases[case]["explanation"]
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        score = ex["score"]
        dcol1.metric(
            "stress score", "n/a" if score is None else f"{score:.3f}"
        )
        dcol2.metric("decision", ex["decision"])
        dcol3.metric("confidence", f"{ex['confidence']:.2f}")
        dcol4.metric(
            "true label", "stress" if cases[case]["true_label"] else "non-stress"
        )
        lo, hi = ex["thresholds"]
        st.caption(
            f"Decision thresholds ({ex['threshold_origin']}): low ≤ {lo}, "
            f"high ≥ {hi}; between -> uncertain. Membership values are NOT "
            "calibrated probabilities."
        )
        st.subheader("Fired rules (strongest first)")
        if ex["fired_rules"]:
            st.table(pd.DataFrame(ex["fired_rules"]).head(10), hide_index=True)
        else:
            st.write("no rule reached non-zero activation")
        if ex["skipped_rules"]:
            st.caption(
                "rules skipped (missing inputs / modality): "
                + ", ".join(ex["skipped_rules"])
            )
        st.subheader("Input memberships")
        mrows = []
        for iname, terms in ex["memberships"].items():
            for t, deg in terms.items():
                mrows.append(
                    {"input": iname, "term": t,
                     "degree": round(float(deg), 3),
                     "value": round(float(ex["inputs"][iname]), 3)}
                )
        st.dataframe(pd.DataFrame(mrows), hide_index=True)
        if bounds:
            st.subheader("Membership bounds used for this fold")
            fb = bounds.get("membership_bounds_per_fold", {}).get(
                "fuzzy_quality_personalized", {}
            ).get(cases[case]["fold"], {})
            st.json(fb)

# --------------------------------------------------------------- export
with tabs[3]:
    st.header("Export results")
    st.write("Download the generated reports and machine-readable artifacts.")
    files = [
        (REP / "data_audit.md", "data audit (markdown)"),
        (REP / "evaluation_report.md", "evaluation report (markdown)"),
        (ART / "data_audit.json", "audit (JSON)"),
        (ART / "evaluation_results.json", "results (JSON)"),
        (ART / "fold_metrics.csv", "fold metrics (CSV)"),
        (ART / "participant_metrics.csv", "participant metrics (CSV)"),
        (ART / "explainability_examples.json", "explanations (JSON)"),
        (ART / "environment.json", "environment + config (JSON)"),
    ]
    found_any = False
    for path, label in files:
        if path.is_file():
            found_any = True
            st.download_button(
                f"⬇ {label}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/plain",
            )
    if not found_any:
        st.info("Nothing to export yet - run the validation and pipeline CLIs.")
    st.divider()
    st.caption(
        "This tool processes data locally only. No data is sent to external "
        "services. Fuzzy outputs are rule-based indices, not medical "
        "assessments."
    )
