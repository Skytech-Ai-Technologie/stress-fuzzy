"""Loading, participant-column normalisation, and window segmentation.

Design rules enforced here:
- The participant-ID alias (``Subject`` vs ``subjet``) is normalised to a
  canonical ``subject`` column while the original column is preserved as-is.
- Windows are cut *within* contiguous same-label phases of one participant, so
  a window never mixes labels and never spans two participants.
- Nothing in this module drops rows silently: dropped labels, tail remainders
  of phases, and duplicate rows are counted and reported by the audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


class DataError(RuntimeError):
    """Raised when a required data condition fails (missing file/column/...)."""


@dataclass
class AlignmentReport:
    """Outcome of the multimodal alignment checks."""

    facial_usable: bool = False
    audio_usable: bool = False
    instant_questionnaires_usable: bool = False
    blockers: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def multimodal_allowed(self) -> bool:
        return self.facial_usable or self.audio_usable

    def to_dict(self) -> dict:
        return {
            "facial_usable": self.facial_usable,
            "audio_usable": self.audio_usable,
            "instant_questionnaires_usable": self.instant_questionnaires_usable,
            "multimodal_allowed": self.multimodal_allowed,
            "blockers": self.blockers,
            "notes": self.notes,
        }


def normalize_subject_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return (df with canonical 'subject' column, alias that was found).

    Raises DataError when no known alias is present. The original column is
    left untouched so provenance stays auditable.
    """
    for alias in C.SUBJECT_ALIASES:
        if alias in df.columns:
            out = df.copy()
            out["subject"] = pd.to_numeric(out[alias], errors="raise").astype("int64")
            return out, alias
    raise DataError(
        f"no participant column found; expected one of {list(C.SUBJECT_ALIASES)}, "
        f"got columns {list(df.columns)}"
    )


def load_physiological(data_dir: Path | str, window_file: str) -> pd.DataFrame:
    """Load one physiological window file with alias normalisation."""
    path = Path(data_dir) / C.PHYSIO_FILES[window_file]
    if not path.is_file():
        raise DataError(
            f"physiological file missing: {path}. Place the extracted WorkStress3D "
            "dataset under the data directory (see README) and re-run."
        )
    df = pd.read_csv(path)
    missing = [c for c in C.SIGNAL_COLUMNS + ["emotion"] if c not in df.columns]
    if missing:
        raise DataError(f"{path.name}: missing expected columns {missing}")
    df, alias = normalize_subject_column(df)
    df.attrs["subject_alias"] = alias
    df.attrs["source_file"] = str(path)
    for col in C.SIGNAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_surveys(data_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Load every survey file that exists; report (not raise) on missing ones."""
    out: dict[str, pd.DataFrame] = {}
    for key, rel in C.SURVEY_FILES.items():
        path = Path(data_dir) / rel
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        try:
            df, alias = normalize_subject_column(df)
            df.attrs["subject_alias"] = alias
        except DataError:
            df.attrs["subject_alias"] = None
        out[key] = df
    return out


def load_facial(data_dir: Path | str) -> pd.DataFrame | None:
    path = Path(data_dir) / C.OTHER_FILES["facial_expression"]
    if not path.is_file():
        return None
    return pd.read_csv(path)


@dataclass
class WindowIndex:
    """Non-overlapping windows into a raw physiological stream.

    ``table`` has one row per window: subject, phase_id, ordinal position,
    label, and the [start, end) row slice into the raw dataframe.
    """

    table: pd.DataFrame
    raw: pd.DataFrame
    window_samples: int
    dropped_tail_rows: int = 0
    dropped_label_rows: int = 0
    label_phase_counts: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.table)

    def window_samples_frame(self, i: int) -> pd.DataFrame:
        row = self.table.iloc[i]
        return self.raw.loc[int(row.start) : int(row.end) - 1, C.SIGNAL_COLUMNS]

    def summarize(self) -> dict:
        return {
            "n_windows": len(self.table),
            "n_participants": int(self.table.subject.nunique()) if len(self.table) else 0,
            "window_samples": self.window_samples,
            "dropped_tail_rows": self.dropped_tail_rows,
            "dropped_label_rows": self.dropped_label_rows,
            "windows_per_participant": (
                self.table.groupby("subject").size().describe().round(2).to_dict()
                if len(self.table)
                else {}
            ),
            "label_counts": (
                self.table.label.value_counts().sort_index().to_dict()
                if len(self.table)
                else {}
            ),
        }


def segment_windows(
    raw: pd.DataFrame,
    window_samples: int,
    drop_labels: tuple = C.DROP_LABELS,
) -> WindowIndex:
    """Cut the stream into windows inside contiguous same-label phases.

    Phases are contiguous runs of one label for one participant. Windows never
    cross a phase boundary, so every window has a pure label. Tail samples of
    a phase that do not fill a whole window are dropped and counted.
    """
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")

    raw = raw.reset_index(drop=True)
    if "subject" not in raw.columns:
        raw, _alias = normalize_subject_column(raw)
    drop_mask = raw["emotion"].isin(list(drop_labels))
    dropped_label_rows = int(drop_mask.sum())

    phase_id = (raw["subject"] != raw["subject"].shift()) | (
        raw["emotion"] != raw["emotion"].shift()
    )
    phase_id = phase_id.cumsum()

    rows = []
    dropped_tail = 0
    kept = raw.loc[~drop_mask]
    kept_phases = phase_id.loc[kept.index]
    for pid, grp in kept.groupby(kept_phases, sort=True):
        subj = int(grp["subject"].iloc[0])
        label = int(grp["emotion"].iloc[0])
        start = int(grp.index[0])
        n = len(grp)
        k = n // window_samples
        dropped_tail += n - k * window_samples
        for j in range(k):
            rows.append(
                {
                    "subject": subj,
                    "phase_id": int(pid),
                    "window": j,
                    "label": label,
                    "start": start + j * window_samples,
                    "end": start + (j + 1) * window_samples,
                }
            )
    table = pd.DataFrame(rows)
    return WindowIndex(
        table=table,
        raw=raw,
        window_samples=window_samples,
        dropped_tail_rows=dropped_tail,
        dropped_label_rows=dropped_label_rows,
    )


def phase_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-participant contiguous label-phase statistics (for the audit)."""
    change = (raw["subject"] != raw["subject"].shift()) | (
        raw["emotion"] != raw["emotion"].shift()
    )
    pid = change.cumsum()
    g = raw.groupby(pid)
    stats = g.agg(
        subject=("subject", "first"),
        emotion=("emotion", "first"),
        length=("emotion", "size"),
        start=("emotion", lambda s: int(s.index[0])),
    ).reset_index(drop=True)
    return stats


def check_alignment(data_dir: Path | str) -> AlignmentReport:
    """Decide whether multimodal fusion may be enabled.

    Alignment must be *demonstrated*, not assumed. Each modality needs a
    participant key and (for temporal joins) a timestamp; WorkStress3D ships
    neither for audio/facial, so fusion stays blocked.
    """
    rep = AlignmentReport()
    data_dir = Path(data_dir)

    facial = load_facial(data_dir)
    if facial is None:
        rep.blockers.append("facial_expression.csv not found; modality unavailable")
    elif not any(a in facial.columns for a in C.SUBJECT_ALIASES):
        rep.blockers.append(
            "facial_expression.csv has no participant column "
            f"(columns: {list(facial.columns)}); images cannot be grouped by "
            "participant, so training or evaluating on them would leak "
            "identities across folds"
        )
    else:
        rep.facial_usable = True  # participant key present; still no timestamps

    audio_dir = data_dir / C.OTHER_FILES["audio_dir"]
    mapping = [
        p
        for p in audio_dir.glob("*")
        if p.is_file()
        and p.suffix.lower() in {".csv", ".json", ".txt", ".xlsx"}
        and any(k in p.name.lower() for k in ("map", "subject", "session", "id"))
    ]
    if not audio_dir.is_dir():
        rep.blockers.append("AudioData/ not found; modality unavailable")
    elif not mapping:
        rep.blockers.append(
            f"{len(list(audio_dir.glob('ses*.m4a')))} audio sessions found but no "
            "session-to-participant mapping file exists; audio cannot be grouped "
            "by participant"
        )
    else:
        rep.audio_usable = True
    test_dir = audio_dir / "test"
    if test_dir.is_dir():
        wavs = sorted(p.name for p in test_dir.glob("*.wav"))
        if wavs:
            rep.notes.append(
                f"AudioData/test/ holds {len(wavs)} wav files with a different "
                "naming scheme (e.g. " + ", ".join(wavs[:3]) + "); provenance is "
                "unclear, so this folder is NOT treated as an independent test set"
            )

    surveys = load_surveys(data_dir)
    iq = surveys.get("instant_questionnaires")
    if iq is None:
        rep.notes.append("instantQuestionnaires.csv not found")
    elif not any(
        k in str(c).lower() for c in iq.columns for k in ("time", "date", "stamp")
    ):
        rep.blockers.append(
            "instantQuestionnaires.csv has no timestamp column; momentary survey "
            "answers cannot be aligned to physiological windows in time"
        )
    else:
        rep.instant_questionnaires_usable = True
    return rep
