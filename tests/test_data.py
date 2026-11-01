"""Tests: participant alias handling, window segmentation, alignment gates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stress_fuzzy import config as C
from stress_fuzzy.data import (
    DataError,
    WindowIndex,
    check_alignment,
    load_physiological,
    normalize_subject_column,
    segment_windows,
)
from conftest import make_raw


def test_alias_subjet_normalized(tmp_path):
    p = tmp_path / "physio.csv"
    make_raw(num_subjects=2, subject_col="subjet").to_csv(p, index=False)
    df = pd.read_csv(p)
    out, alias = normalize_subject_column(df)
    assert alias == "subjet"
    assert "subject" in out.columns
    assert out.subject.tolist() == df["subjet"].tolist()
    assert "subjet" in out.columns  # original column preserved


def test_alias_missing_raises():
    with pytest.raises(DataError):
        normalize_subject_column(pd.DataFrame({"eda": [1.0]}))


def test_load_physiological_missing_file(tmp_path):
    with pytest.raises(DataError, match="missing"):
        load_physiological(tmp_path, "15sn")


def test_load_physiological_alias(tmp_path):
    d = tmp_path / "PhysiologicalSignals"
    d.mkdir()
    make_raw(num_subjects=2, subject_col="subjet").to_csv(
        d / "physiological_signals_30sn.csv", index=False
    )
    df = load_physiological(tmp_path, "30sn")
    assert df.attrs["subject_alias"] == "subjet"
    assert set(df.subject.unique()) == {1, 2}


def test_segment_windows_pure_labels_and_no_crossing(synthetic_raw):
    raw = synthetic_raw
    widx = segment_windows(raw, C.DEFAULT_WINDOW_SAMPLES)
    assert len(widx) > 0
    # every window covers exactly window_samples rows of one subject and label
    for i in range(len(widx.table)):
        row = widx.table.iloc[i]
        sl = widx.raw.loc[int(row.start): int(row.end) - 1]
        assert len(sl) == C.DEFAULT_WINDOW_SAMPLES
        assert sl.subject.nunique() == 1
        assert sl.emotion.nunique() == 1
        assert int(sl.emotion.iloc[0]) == int(row.label)


def test_segment_windows_drops_ambiguous_label():
    raw = make_raw(num_subjects=1)
    raw.loc[raw.index[: C.DEFAULT_WINDOW_SAMPLES], "emotion"] = 2
    widx = segment_windows(raw, C.DEFAULT_WINDOW_SAMPLES)
    assert 2 not in set(widx.table.label.unique())
    assert widx.dropped_label_rows == C.DEFAULT_WINDOW_SAMPLES


def test_segment_windows_counts_tail():
    raw = make_raw(num_subjects=1)  # phases of 56, 112, 56, 112
    raw = raw.head(56 + 90)         # second phase truncated to 90 -> tail 34
    widx = segment_windows(raw, 56)
    first_subject_windows = widx.table
    assert len(first_subject_windows) == 2
    assert widx.dropped_tail_rows == 34


def test_check_alignment_blocks_synthetic(synthetic_data_dir):
    rep = check_alignment(synthetic_data_dir)
    assert not rep.multimodal_allowed
    assert rep.blockers, "expected alignment blockers without participant keys"
    assert any("facial" in b or "participant" in b for b in rep.blockers)


def test_check_alignment_test_dir_note(synthetic_data_dir):
    test_dir = synthetic_data_dir / "AudioData" / "test"
    test_dir.mkdir()
    (test_dir / "ses_n01.wav").write_bytes(b"x")
    rep = check_alignment(synthetic_data_dir)
    assert any("test" in n for n in rep.notes)
