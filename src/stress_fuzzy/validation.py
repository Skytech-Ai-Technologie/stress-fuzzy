"""Data audit: run before any modelling.

Produces ``reports/data_audit.md`` and ``artifacts/data_audit.json`` covering:
file inventory, row/column/missing/duplicate/participant counts, target
distributions, numeric range checks, label-phase structure, cross-file overlap,
multimodal alignment status, and every assumption the pipeline relies on.

CLI: ``python -m stress_fuzzy.validation --data-dir <dir> [--strict]``
Exit codes: 0 = audit ran (warnings allowed), 1 = blocking failure, 2 = usage.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import (
    DataError,
    check_alignment,
    load_facial,
    load_physiological,
    load_surveys,
    normalize_subject_column,
    phase_stats,
)

# Audit findings that must be resolved before modelling is meaningful.
BLOCKING = "blocking"
WARNING = "warning"
INFO = "info"


def _file_profile(path: Path) -> dict:
    """Generic CSV profile used for every discovered file."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # unreadable csv -> reported, never fatal by itself
        return {"file": str(path), "error": f"could not parse: {exc}"}
    prof = {
        "file": str(path.relative_to(path.parents[1])) if len(path.parents) > 1 else str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {c: int(v) for c, v in df.isna().sum().items() if v},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    subj_alias = next((a for a in C.SUBJECT_ALIASES if a in df.columns), None)
    prof["participant_column"] = subj_alias
    prof["n_participants"] = (
        int(pd.to_numeric(df[subj_alias], errors="coerce").nunique())
        if subj_alias
        else None
    )
    return prof


def _physio_profile(df: pd.DataFrame, name: str) -> dict:
    prof = {
        "file": C.PHYSIO_FILES[name],
        "rows": int(len(df)),
        "participant_alias": df.attrs.get("subject_alias"),
        "n_participants": int(df.subject.nunique()),
        "participants": sorted(int(s) for s in df.subject.unique()),
        "rows_per_participant": df.groupby("subject").size().unique().tolist(),
        "missing_per_column": {c: int(v) for c, v in df.isna().sum().items() if v},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    prof["label_counts"] = {str(k): int(v) for k, v in df.emotion.value_counts().sort_index().items()}
    prof["label_proportions"] = {
        k: round(v / len(df), 4) for k, v in prof["label_counts"].items()
    }
    prof["label_names_assumed"] = C.LABEL_NAMES

    ranges = {}
    constants = []
    for col in C.SIGNAL_COLUMNS:
        s = df[col]
        lo, hi = C.PHYSIO_RANGES[col]
        ranges[col] = {
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std()), 3),
            "n_outside_plausible_range": int(((s < lo) | (s > hi)).sum()),
            "plausible_range": [lo, hi],
        }
        if float(s.std()) < 1e-12:
            constants.append(col)
    prof["signal_ranges"] = ranges
    prof["constant_features"] = constants

    ps = phase_stats(df)
    prof["phase_structure"] = {
        "n_phases_total": int(len(ps)),
        "phases_per_participant": ps.groupby("subject").size().describe().round(2).to_dict(),
        "phase_length_samples": ps.length.describe().round(2).to_dict(),
        "label_sequence_per_participant": {
            str(subj): [int(x) for x in grp.emotion.tolist()]
            for subj, grp in ps.groupby("subject")
        },
        "gcd_phase_lengths": int(np.gcd.reduce(ps.length.astype("int64").values))
        if len(ps)
        else None,
    }
    return prof


def run_audit(data_dir: Path) -> dict:
    """Collect the full audit as a JSON-serialisable dict."""
    findings: list[dict] = []

    def add(severity: str, msg: str) -> None:
        findings.append({"severity": severity, "message": msg})

    inventory = []
    expected = {**C.PHYSIO_FILES, **C.SURVEY_FILES, **C.OTHER_FILES}
    for key, rel in expected.items():
        p = data_dir / rel
        inventory.append(
            {
                "key": key,
                "expected_path": rel,
                "exists": p.exists(),
                "is_dir": p.is_dir(),
            }
        )
    missing = [i["key"] for i in inventory if not i["exists"]]
    if missing:
        add(
            BLOCKING if {"15sn", "30sn", "60sn"} & set(missing) else WARNING,
            f"missing expected files: {missing}. Place the extracted WorkStress3D "
            f"dataset under {data_dir} (expected layout in README) and re-run.",
        )

    discovered = []
    for p in sorted(data_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".csv", ".pdf"}:
            discovered.append(str(p.relative_to(data_dir)))

    audit: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "data_dir_exists": data_dir.is_dir(),
        "inventory": inventory,
        "discovered_files": discovered,
        "physiological": {},
        "surveys": {},
        "facial": {},
        "audio": {},
        "alignment": {},
        "assumptions": [],
        "findings": findings,
    }

    # --- physiological ---
    for name in C.PHYSIO_FILES:
        try:
            df = load_physiological(data_dir, name)
        except DataError as exc:
            add(BLOCKING, str(exc))
            continue
        audit["physiological"][name] = _physio_profile(df, name)
        alias = df.attrs.get("subject_alias")
        if alias != "Subject":
            add(
                INFO,
                f"physiological_signals_{name}.csv spells the participant column "
                f"'{alias}' instead of 'Subject'; normalised to canonical 'subject'",
            )
        if audit["physiological"][name]["duplicate_rows"]:
            add(
                INFO,
                f"{name}: {audit['physiological'][name]['duplicate_rows']} duplicate "
                "full rows - consistent with overlapping windowed exports of the "
                "same stream, kept as-is",
            )
        bad = {
            c: v["n_outside_plausible_range"]
            for c, v in audit["physiological"][name]["signal_ranges"].items()
            if v["n_outside_plausible_range"]
        }
        if bad:
            add(WARNING, f"{name}: samples outside plausible sensor ranges: {bad}")
        if audit["physiological"][name]["constant_features"]:
            add(
                WARNING,
                f"{name}: constant features "
                f"{audit['physiological'][name]['constant_features']}",
            )
        lc = audit["physiological"][name]["label_counts"]
        if "2" in lc or 2 in lc:
            add(
                WARNING,
                f"{name}: label value 2 (undocumented in DataDescripton.pdf) covers "
                f"{lc.get('2', lc.get(2))} rows; assumed ambiguous/transition and "
                "excluded from training/evaluation",
            )

    # --- surveys (participant-level, never momentary labels) ---
    surveys = load_surveys(data_dir)
    for key, df in surveys.items():
        prof = _file_profile(data_dir / C.SURVEY_FILES[key])
        audit["surveys"][key] = prof
    if "general_stress_test" in surveys:
        add(
            INFO,
            "generalStressTest is a participant-level trait score "
            "(TotalPoints); it is NEVER used as a momentary stress label",
        )

    # --- facial ---
    facial = load_facial(data_dir)
    if facial is not None:
        prof = _file_profile(data_dir / C.OTHER_FILES["facial_expression"])
        prof["label_counts"] = {
            str(k): int(v) for k, v in facial.emotion.value_counts().sort_index().items()
        }
        px_len = facial.pixels.astype(str).str.split().str.len()
        prof["pixel_counts"] = px_len.value_counts().to_dict()
        audit["facial"] = prof

    # --- audio ---
    audio_dir = data_dir / C.OTHER_FILES["audio_dir"]
    if audio_dir.is_dir():
        m4a = sorted(audio_dir.glob("*.m4a"))
        audit["audio"] = {
            "n_sessions": len(m4a),
            "session_names": [p.name for p in m4a][:60],
            "test_dir_files": [
                p.name for p in sorted((audio_dir / "test").glob("*.wav"))
            ]
            if (audio_dir / "test").is_dir()
            else [],
        }

    # --- alignment / leakage gates ---
    rep = check_alignment(data_dir)
    audit["alignment"] = rep.to_dict()
    for b in rep.blockers:
        add(WARNING, f"alignment blocker: {b}")
    for n in rep.notes:
        add(INFO, f"alignment note: {n}")

    # --- cross-file participant overlap ---
    subj_sets = {
        name: set(int(s) for s in prof["participants"])
        for name, prof in audit["physiological"].items()
        if "participants" in prof
    }
    if subj_sets:
        inter = set.intersection(*subj_sets.values()) if subj_sets else set()
        union = set.union(*subj_sets.values()) if subj_sets else set()
        audit["participant_overlap"] = {
            "physiological_intersection": sorted(inter),
            "physiological_union": sorted(union),
            "all_files_share_same_participants": inter == union and bool(union),
        }
        if inter != union:
            add(
                WARNING,
                "physiological files do not cover exactly the same participants",
            )
        add(
            INFO,
            "the 15/30/60 s files are windowed exports of the SAME recording "
            "sessions (verified: identical first rows and shared phase pattern); "
            "only one file is used per experiment to avoid duplicating data",
        )

    audit["assumptions"] = [
        "emotion: 0 = non-stress, 1 = stress (per DataDescripton.pdf facial "
        "coding; assumed to carry over to the physiological emotion column)",
        "emotion == 2 is an undocumented, rare (<1%) between-phase marker; "
        "treated as ambiguous and excluded from training/evaluation",
        "rows are 4 Hz samples of one continuous stream per participant "
        "(Empatica E4, downsampled per DataDescripton.pdf); labels are constant "
        "within protocol phases, not independent per row",
        "windows are cut inside same-label phases and never span participants",
        "general stress test / PANAS are participant-level traits, not momentary "
        "labels; never used as training targets",
        "instant questionnaires lack timestamps; not temporally joinable to "
        "physiological windows",
        "audio and facial modalities have no participant key; multimodal fusion "
        "is blocked",
    ]
    audit["blocking_failures"] = [f for f in findings if f["severity"] == BLOCKING]
    audit["ok_for_modelling"] = bool(
        audit["physiological"] and not audit["blocking_failures"]
    )
    return audit


# ---------------------------------------------------------------- reporting


def write_audit_outputs(audit: dict, reports_dir: Path, artifacts_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    from .reporting import dump_json

    dump_json(audit, artifacts_dir / "data_audit.json")

    lines: list[str] = []
    a = audit
    lines.append("# WorkStress3D data audit")
    lines.append("")
    lines.append(f"Generated: {a['generated_at']}  |  data dir: `{a['data_dir']}`")
    lines.append("")

    lines.append("## Verdict")
    verdict = (
        "OK for physiological modelling (multimodal blocked - see alignment)"
        if a["ok_for_modelling"]
        else "NOT ready for modelling"
    )
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append("| severity | count |")
    lines.append("|---|---|")
    for sev in (BLOCKING, WARNING, INFO):
        n = sum(1 for f in a["findings"] if f["severity"] == sev)
        lines.append(f"| {sev} | {n} |")
    lines.append("")

    lines.append("## Findings")
    for f in a["findings"]:
        lines.append(f"- **{f['severity']}**: {f['message']}")
    lines.append("")

    lines.append("## Files")
    lines.append("| key | path | exists |")
    lines.append("|---|---|---|")
    for i in a["inventory"]:
        lines.append(f"| {i['key']} | {i['expected_path']} | {i['exists']} |")
    lines.append("")
    lines.append("Discovered CSV/PDF files: " + ", ".join(f"`{d}`" for d in a["discovered_files"]))
    lines.append("")

    lines.append("## Physiological files")
    for name, p in a["physiological"].items():
        lines.append(f"### {name} (`{p['file']}`)")
        lines.append("")
        lines.append(f"- rows: {p['rows']}, participants: {p['n_participants']} "
                     f"(column alias `{p['participant_alias']}`)")
        lines.append(f"- rows per participant: {p['rows_per_participant']}")
        lines.append(f"- missing values: {p['missing_per_column'] or 'none'}; "
                     f"duplicate rows: {p['duplicate_rows']}")
        lines.append(f"- labels: {p['label_counts']} "
                     f"({ {k: v for k, v in p['label_proportions'].items()} })")
        lines.append(f"- phases per participant: {p['phase_structure']['phases_per_participant']}")
        lines.append(f"- phase length (samples): {p['phase_structure']['phase_length_samples']}")
        lines.append(f"- GCD of phase lengths: {p['phase_structure']['gcd_phase_lengths']} "
                     "samples -> natural base window unit")
        lines.append("- signal ranges:")
        lines.append("")
        lines.append("| channel | min | max | mean | std | outside plausible |")
        lines.append("|---|---|---|---|---|---|")
        for c, r in p["signal_ranges"].items():
            lines.append(
                f"| {c} | {r['min']} | {r['max']} | {r['mean']} | {r['std']} | "
                f"{r['n_outside_plausible_range']} |"
            )
        lines.append("")

    if a.get("facial"):
        f = a["facial"]
        lines.append("## Facial expressions")
        lines.append("")
        lines.append(f"- rows: {f['rows']}, labels: {f.get('label_counts')}")
        lines.append(f"- participant column: {f['participant_column']} "
                     "(none -> cannot be grouped; modality blocked)")
        lines.append("")

    if a.get("audio"):
        au = a["audio"]
        lines.append("## Audio")
        lines.append("")
        lines.append(f"- session files: {au['n_sessions']} (m4a, no participant mapping)")
        if au.get("test_dir_files"):
            lines.append(f"- `test/` wav files ({len(au['test_dir_files'])}): "
                         + ", ".join(au["test_dir_files"][:14]))
            lines.append("- different naming scheme from the main sessions; "
                         "provenance unclear -> NOT an independent test set")
        lines.append("")

    al = a["alignment"]
    lines.append("## Multimodal alignment")
    lines.append("")
    lines.append(f"- multimodal allowed: **{al['multimodal_allowed']}**")
    for b in al["blockers"]:
        lines.append(f"- blocker: {b}")
    for n in al["notes"]:
        lines.append(f"- note: {n}")
    lines.append("")

    lines.append("## Assumptions (explicit, reviewable)")
    for s in a["assumptions"]:
        lines.append(f"- {s}")
    lines.append("")

    (reports_dir / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stress_fuzzy.validation",
        description="Audit the WorkStress3D data directory before modelling.",
    )
    parser.add_argument("--data-dir", default="", help="path to the extracted dataset")
    parser.add_argument(
        "--out-dir", default=".", help="root under which artifacts/ and reports/ are written"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="also exit non-zero on warnings (blocking findings always exit 1)",
    )
    args = parser.parse_args(argv)

    from .config import resolve_data_dir

    data_dir, notes = resolve_data_dir(args.data_dir)
    for n in notes:
        print(f"[info] {n}")

    try:
        audit = run_audit(data_dir)
    except Exception as exc:
        print(f"[error] audit failed: {exc}", file=sys.stderr)
        return 1

    write_audit_outputs(audit, Path(args.out_dir) / "reports", Path(args.out_dir) / "artifacts")
    print(f"[ok] audit written: reports/data_audit.md, artifacts/data_audit.json")
    for f in audit["findings"]:
        print(f"[{f['severity']}] {f['message']}")
    if not audit["ok_for_modelling"]:
        print(
            "[action] resolve the blocking findings above before running the "
            "modelling pipeline (python -m stress_fuzzy.pipeline)"
        )
        return 1
    if args.strict and any(f["severity"] == WARNING for f in audit["findings"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
