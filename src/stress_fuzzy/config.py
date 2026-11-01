"""Central configuration for the stress_fuzzy pipeline.

Every modelling decision that a reader could reasonably question lives here or
in the fuzzy rule file (``stress_fuzzy/rules/fuzzy_rules.yaml``) so that it can
be inspected, cited, and changed without touching implementation code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Candidate locations for the extracted WorkStress3D dataset, in resolution
# order when --data-dir is not given explicitly. The brief documents
# ``data/WorkStress3D``; the user's machine currently has the extraction under
# ``database/``, so both are supported and the resolved path is always reported.
DATA_DIR_CANDIDATES = ("data/WorkStress3D", "database")

PHYSIO_FILES = {
    "15sn": "PhysiologicalSignals/physiological_signals_15sn.csv",
    "30sn": "PhysiologicalSignals/physiological_signals_30sn.csv",
    "60sn": "PhysiologicalSignals/physiological_signals_60sn.csv",
}
SURVEY_FILES = {
    "instant_questionnaires": "Survey/instantQuestionnaires.csv",
    "general_stress_test": "Survey/generalStressTest.csv",
    "panas": "Survey/PANAS.csv",
    "demographic": "Survey/demographic.csv",
}
OTHER_FILES = {
    "facial_expression": "TheFacialExpressions/facial_expression.csv",
    "audio_dir": "AudioData",
}

# Canonical participant column. The dataset spells it ``Subject`` in most files
# and ``subjet`` in physiological_signals_30sn.csv (verified 2026-10-24).
SUBJECT_ALIASES = ("Subject", "subjet", "subject", "Participant")

LABEL_NAMES = {0: "non-stress", 1: "stress", 2: "ambiguous"}

# ``emotion`` == 2 is undocumented in DataDescripton.pdf, appears in 4/20
# participants as a single short block between phases, and covers 0.57% of
# rows. Assumption: transition/ambiguous marker -> excluded from training and
# evaluation. Overridable via Config.drop_labels.
DROP_LABELS = (2,)

# Empatica E4 raw-value plausibility ranges used for quality flags only
# (rows outside a range are never silently deleted, only flagged).
PHYSIO_RANGES = {
    "eda": (0.0, 100.0),        # microsiemens
    "bvp": (-3000.0, 3000.0),   # raw arbitrary units
    "temp": (20.0, 45.0),       # degrees Celsius, wrist skin temperature
    "x": (-128.0, 128.0),       # Empatica E4 accelerometer raw range
    "y": (-128.0, 128.0),
    "z": (-128.0, 128.0),
}
SIGNAL_COLUMNS = ["eda", "bvp", "temp", "x", "y", "z"]
ACC_COLUMNS = ["x", "y", "z"]

SAMPLING_RATE_HZ = 4.0

# Default physiological window. The 15sn file's label runs are exact multiples
# of 56 samples (~14 s at 4 Hz), so 56 is the dataset's own base window unit
# for that file: non-overlapping windows then never straddle label phases.
DEFAULT_WINDOW_FILE = "15sn"
DEFAULT_WINDOW_SAMPLES = 56

RANDOM_SEED = 42


@dataclass
class Config:
    """Runtime configuration; serialisable to JSON for reproducibility."""

    data_dir: str = ""
    window_file: str = DEFAULT_WINDOW_FILE
    window_samples: int = DEFAULT_WINDOW_SAMPLES
    protocol: str = "loso"            # "loso" | "gk5"
    seed: int = RANDOM_SEED
    personalize: bool = True          # per-participant input normalisation (leakage-safe)
    quality_aware: bool = True        # enable quality/motion rules in the fuzzy model
    drop_labels: tuple = DROP_LABELS
    out_dir: str = "."                # root for artifacts/ and reports/
    rules_path: str = ""              # optional override of the bundled YAML rules
    threshold_mode: str = "train_fit"  # "train_fit" | "fixed"
    fixed_thresholds: tuple = (0.35, 0.65)  # used when threshold_mode == "fixed"
    confidence_tau: float = 0.20      # below -> decision "uncertain"
    quality_tau: float = 0.50         # window quality below -> quality warning
    group_kfold_splits: int = 5
    allow_multimodal: bool = False    # fusion stays disabled until alignment is proven
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["drop_labels"] = list(self.drop_labels)
        d["fixed_thresholds"] = list(self.fixed_thresholds)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.out_dir) / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return Path(self.out_dir) / "reports"


def resolve_data_dir(explicit: str = "") -> tuple[Path, list]:
    """Return (resolved data dir or the first candidate if none exists, notes).

    The returned path may not exist; callers must check and report cleanly.
    """
    notes: list = []
    if explicit:
        candidates = [explicit]
    else:
        candidates = list(DATA_DIR_CANDIDATES)
        notes.append(
            "--data-dir not given; probed candidates: " + ", ".join(candidates)
        )
    for cand in candidates:
        p = Path(cand)
        if p.is_dir():
            notes.append(f"data dir resolved to {p.resolve()}")
            return p, notes
    notes.append(
        f"no data directory found (probed: {', '.join(candidates)}); "
        "place the extracted WorkStress3D files under one of these paths "
        "or pass --data-dir explicitly"
    )
    return Path(candidates[0]), notes
