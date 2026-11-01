"""Tests: audit behaviour on synthetic dirs, report generation, pipeline CLI.
Numbers from these fixtures are software tests only, never scientific results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stress_fuzzy import config as C
from stress_fuzzy.validation import main as audit_main, run_audit, write_audit_outputs
from stress_fuzzy.pipeline import main as pipeline_main


def test_audit_ok_on_synthetic_dir(synthetic_data_dir, tmp_path):
    audit = run_audit(synthetic_data_dir)
    assert audit["ok_for_modelling"]
    assert audit["physiological"]["15sn"]["n_participants"] == 6
    # subjet alias in the 30sn file is detected and reported
    assert audit["physiological"]["30sn"]["participant_alias"] == "subjet"
    assert any("subjet" in f["message"] for f in audit["findings"])
    # multimodal blocked
    assert not audit["alignment"]["multimodal_allowed"]


def test_audit_missing_files_reported_cleanly(tmp_path):
    empty = tmp_path / "nowhere"
    empty.mkdir()
    audit = run_audit(empty)
    assert not audit["ok_for_modelling"]
    assert audit["blocking_failures"]
    assert any("missing expected files" in f["message"] for f in audit["findings"])


def test_audit_writes_outputs(synthetic_data_dir, tmp_path):
    audit = run_audit(synthetic_data_dir)
    write_audit_outputs(audit, tmp_path / "reports", tmp_path / "artifacts")
    md = (tmp_path / "reports" / "data_audit.md").read_text(encoding="utf-8")
    js = json.loads((tmp_path / "artifacts" / "data_audit.json").read_text())
    assert "data audit" in md
    assert "Assumptions" in md
    assert js["ok_for_modelling"] is True


def test_audit_cli_exit_codes(synthetic_data_dir, tmp_path, capsys):
    rc = audit_main(
        ["--data-dir", str(synthetic_data_dir), "--out-dir", str(tmp_path)]
    )
    assert rc == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = audit_main(["--data-dir", str(empty), "--out-dir", str(tmp_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "blocking" in out.lower()


def test_pipeline_end_to_end_synthetic(synthetic_data_dir, tmp_path, capsys):
    rc = pipeline_main(
        [
            "--data-dir", str(synthetic_data_dir),
            "--out-dir", str(tmp_path),
            "--protocol", "gk5",
        ]
    )
    assert rc == 0
    # machine-readable outputs exist and parse
    for name in (
        "evaluation_results.json",
        "fold_metrics.csv",
        "participant_metrics.csv",
        "fuzzy_membership_bounds.json",
        "explainability_examples.json",
        "environment.json",
    ):
        assert (tmp_path / "artifacts" / name).is_file(), name
    res = json.loads(
        (tmp_path / "artifacts" / "evaluation_results.json").read_text()
    )
    assert res["protocol"] == "gk5"
    assert res["n_participants"] == 6
    assert "fuzzy_quality_personalized" in res["models"]
    # human report contains key sections
    md = (tmp_path / "reports" / "evaluation_report.md").read_text(encoding="utf-8")
    for section in (
        "Model comparison",
        "Coverage vs performance",
        "Ablations",
        "Confusion matrices",
        "Known limitations",
    ):
        assert section in md, section
    # explanations are held-out and explainable
    ex = json.loads(
        (tmp_path / "artifacts" / "explainability_examples.json").read_text()
    )
    assert ex and all(e["held_out"] for e in ex)
    assert all("fired_rules" in e["explanation"] for e in ex)
    # environment snapshot carries versions + config
    env = json.loads((tmp_path / "artifacts" / "environment.json").read_text())
    assert "scikit-learn" in env["versions"]
    assert env["config"]["protocol"] == "gk5"


def test_pipeline_multimodal_refused_without_alignment(synthetic_data_dir, tmp_path, capsys):
    rc = pipeline_main(
        [
            "--data-dir", str(synthetic_data_dir),
            "--out-dir", str(tmp_path),
            "--allow-multimodal",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "alignment" in err.lower()


def test_synthetic_results_never_claimed_as_real(tmp_path):
    """Guard: pipeline output must always carry the exploratory disclaimer."""
    md = (tmp_path / "reports" / "evaluation_report.md")
    if md.exists():
        text = md.read_text(encoding="utf-8")
        assert "exploratory" in text
