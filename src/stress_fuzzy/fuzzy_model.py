"""Transparent quality-aware Mamdani fuzzy inference for stress scoring.

Why Mamdani (and not Sugeno): Mamdani consequents are linguistic terms
(low/medium/high stress), so every fired rule can be verbalised directly to a
user ("if EDA is high and motion is low and quality is good then stress is
high"). Sugeno consequents are linear functions - more compact numerically but
less readable, and readability is the point of this model.

Implementation: a small, fully unit-tested inference engine (triangle and
trapezoid membership functions, min for AND, max aggregation, centroid
defuzzification over the output universe). No external fuzzy framework is
used, so behaviour is exactly specified by these functions and the rule YAML.

 leakage-safety contract: ``fit`` uses training data only (membership
 bounds from training quantiles, thresholds from training scores). Personal
normalisation is applied to the *inputs* before fitting/prediction and uses
only each participant's own windows (never labels, never other participants).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEFAULT_RULES_FILE = Path(__file__).parent / "rules" / "fuzzy_rules.yaml"

TERM_LEVELS = ("low", "medium", "high")
UNCERTAIN = "uncertain"
LABEL_FOR_DECISION = {"low": 0, "high": 1}


# ------------------------------------------------------------------ MFs


def triangle(x, a: float, b: float, c: float) -> np.ndarray | float:
    """Triangular membership function; 0 outside [a, c], 1 at b."""
    x = np.asarray(x, dtype=float)
    if a == b == c:
        return (x == b).astype(float)
    out = np.zeros_like(x, dtype=float)
    if b != a:
        left = (x - a) / (b - a)
        out = np.where(x <= b, np.maximum(out, left), out)
    if c != b:
        right = (c - x) / (c - b)
        out = np.where(x > b, np.maximum(out, right), out)
    out = np.where(x < a, 0.0, out)
    out = np.where(x > c, 0.0, out)
    out = np.clip(out, 0.0, 1.0)
    return float(out) if out.ndim == 0 else out


def trapezoid(x, a: float, b: float, c: float, d: float) -> np.ndarray | float:
    """Trapezoidal membership function; 1 on [b, c].

    Shoulder behaviour: when a == b (left shoulder) the function saturates
    at 1 for all x <= a; when c == d (right shoulder) it saturates at 1 for
    all x >= d. This keeps extreme-but-valid inputs fully inside the outer
    terms of a fitted partition instead of falling out of every term.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    if b != a:
        out = np.maximum(out, (x - a) / (b - a))
    if d != c:
        out = np.maximum(out, (d - x) / (d - c))
    out = np.where((x >= b) & (x <= c), 1.0, out)
    if a == b:
        out = np.where(x <= a, 1.0, out)
    else:
        out = np.where(x < a, 0.0, out)
    if c == d:
        out = np.where(x >= c, 1.0, out)
    else:
        out = np.where(x > d, 0.0, out)
    return float(np.clip(out, 0.0, 1.0)) if out.ndim == 0 else np.clip(out, 0.0, 1.0)


def apply_mf(x, spec: dict):
    """Dispatch on spec {"type": triangle|trapezoid, "params": [...]}."""
    params = [float(p) for p in spec["params"]]
    if spec["type"] == "triangle":
        return triangle(x, *params)
    if spec["type"] == "trapezoid":
        return trapezoid(x, *params)
    raise ValueError(f"unknown MF type {spec['type']!r}")


def partition_3(spec: dict, x_lo: float, m1: float, m2: float, m3: float, x_hi: float) -> dict:
    """Shoulder-trapezoid 3-term partition of [x_lo, x_hi] around m1<m2<m3.

    Used for training-fitted inputs: low/medium/high with 50%-overlap
    trapezoids, saturating at the domain edges so extreme (but valid) values
    stay fully inside the outer terms.
    """
    w = max(m3 - m1, 1e-9)
    lo = min(x_lo, m1 - w)
    hi = max(x_hi, m3 + w)
    return {
        "low": {"type": "trapezoid", "params": [lo, lo, m1, m2]},
        "medium": {"type": "triangle", "params": [m1, m2, m3]},
        "high": {"type": "trapezoid", "params": [m2, m3, hi, hi]},
        "fit": {"lo": lo, "m1": m1, "m2": m2, "m3": m3, "hi": hi},
    }


# ------------------------------------------------------------- rule base


@dataclass
class Rule:
    id: str
    antecedents: dict   # input name -> term name
    consequent: str     # low | medium | high | uncertain
    description: str = ""


def load_rules(path: Path | str | None = None) -> dict:
    p = Path(path) if path else DEFAULT_RULES_FILE
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_rules(rule_spec: list[dict]) -> list[Rule]:
    rules = []
    for r in rule_spec:
        rules.append(
            Rule(
                id=r["id"],
                antecedents=dict(r["if"]),
                consequent=str(r["then"]),
                description=r.get("description", ""),
            )
        )
    return rules


# ---------------------------------------------------------------- engine


@dataclass
class FuzzyOutput:
    score: float | None
    decision: str            # "low" | "uncertain" | "high"
    forced_label: int        # 0/1 forced by 0.5 threshold (baseline comparability)
    confidence: float
    stress_mass: float
    uncertain_mass: float
    memberships: dict        # input -> {term: degree}
    fired_rules: list        # [{id, description, strength, consequent}]
    skipped_rules: list      # rule ids skipped for missing inputs
    quality: float | None


class FuzzyEngine:
    """Stateless Mamdani engine over a fixed membership configuration."""

    def __init__(self, input_terms: dict, output_terms: dict, rules: list[Rule],
                 output_grid: np.ndarray | None = None):
        self.input_terms = input_terms      # input -> {term: mf_spec}
        self.output_terms = output_terms
        self.rules = rules
        self.grid = output_grid if output_grid is not None else np.linspace(0.0, 1.0, 101)
        self._output_mfs = {
            name: apply_mf(self.grid, spec) for name, spec in output_terms.items()
        }

    def fuzzify(self, inputs: dict) -> dict:
        out = {}
        for name, value in inputs.items():
            terms = self.input_terms.get(name)
            if terms is None or value is None or not np.isfinite(value):
                out[name] = {} if terms is None else {
                    t: 0.0 for t in terms
                }
                continue
            out[name] = {t: float(apply_mf(value, spec)) for t, spec in terms.items()}
        return out

    def infer(self, inputs: dict) -> FuzzyOutput:
        memberships = self.fuzzify(inputs)
        aggregated = np.zeros_like(self.grid)
        stress_mass = 0.0
        uncertain_mass = 0.0
        fired, skipped = [], []

        for rule in self.rules:
            degrees = []
            missing = False
            for iname, term in rule.antecedents.items():
                if iname not in inputs or iname not in self.input_terms:
                    missing = True
                    break
                deg = memberships.get(iname, {}).get(term)
                if deg is None:
                    missing = True
                    break
                degrees.append(deg)
            if missing:
                skipped.append(rule.id)
                continue
            alpha = min(degrees) if degrees else 0.0
            if alpha <= 0:
                continue
            if rule.consequent == UNCERTAIN:
                uncertain_mass = max(uncertain_mass, alpha)
                fired.append(
                    {"id": rule.id, "description": rule.description,
                     "strength": round(alpha, 4), "consequent": "uncertain"}
                )
            else:
                stress_mass = max(stress_mass, alpha)
                clipped = np.minimum(self._output_mfs[rule.consequent], alpha)
                aggregated = np.maximum(aggregated, clipped)
                fired.append(
                    {"id": rule.id, "description": rule.description,
                     "strength": round(alpha, 4), "consequent": rule.consequent}
                )

        if aggregated.max() > 0:
            score = float(
                np.sum(self.grid * aggregated) / np.sum(aggregated)
            )
        else:
            score = None

        quality = inputs.get("quality")
        confidence = stress_mass * (quality if quality is not None and np.isfinite(quality) else 1.0)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return FuzzyOutput(
            score=score,
            decision=UNCERTAIN,       # refined by the classifier thresholds
            forced_label=0 if (score is None or score < 0.5) else 1,
            confidence=confidence,
            stress_mass=float(stress_mass),
            uncertain_mass=float(uncertain_mass),
            memberships=memberships,
            fired_rules=sorted(fired, key=lambda r: -r["strength"]),
            skipped_rules=skipped,
            quality=quality,
        )


# ---------------------------------------------------- classifier wrapper

INPUT_ORDER = ["eda_level", "bvp_variability", "temp_deviation", "motion", "quality"]


def build_fuzzy_inputs(features: pd.DataFrame, personalize: bool = True) -> pd.DataFrame:
    """Map window features to fuzzy inputs, optionally per-participant
    normalised (median/MAD robust z-score using the participant's OWN windows
    only - no labels, no other participants; safe under grouped splits).
    """
    cols = {
        "eda_level": features["eda__mean"].astype(float),
        "bvp_variability": features["bvp__std"].astype(float),
        "temp_deviation": features["temp_deviation"].astype(float),
        "motion": features["acc_magnitude_std"].astype(float),
        "quality": features["quality"].astype(float).clip(0.0, 1.0),
    }
    out = pd.DataFrame(cols, index=features.index)
    if personalize:
        subjects = features["subject"].values
        for col in ("eda_level", "bvp_variability", "temp_deviation", "motion"):
            v = out[col].to_numpy(float)
            normed = np.empty_like(v)
            for subj in np.unique(subjects):
                m = subjects == subj
                med = np.nanmedian(v[m])
                mad = np.nanmedian(np.abs(v[m] - med))
                scale = 1.4826 * mad if mad > 1e-9 else (
                    np.nanstd(v[m]) if np.nanstd(v[m]) > 1e-9 else 1.0
                )
                normed[m] = (v[m] - med) / scale
            out[col] = normed
    return out[INPUT_ORDER]


class FuzzyStressClassifier:
    """Fit/predict wrapper with training-only bounds and threshold selection."""

    def __init__(self, rules_path: Path | str | None = None, use_quality: bool = True,
                 threshold_mode: str = "train_fit",
                 fixed_thresholds: tuple = (0.35, 0.65),
                 confidence_tau: float = 0.20):
        self.rules_path = str(rules_path) if rules_path else None
        spec = load_rules(rules_path)
        self.spec = spec
        self.use_quality = use_quality
        self.threshold_mode = threshold_mode
        self.fixed_thresholds = tuple(fixed_thresholds)
        self.confidence_tau = confidence_tau

        all_rules = parse_rules(spec["rules"])
        if use_quality:
            self.rules = all_rules
        else:
            # Ablation: drop the quality/uncertainty rules and the quality
            # antecedent from remaining rules; quality is then assumed good.
            self.rules = []
            for r in all_rules:
                if r.consequent == UNCERTAIN:
                    continue
                ants = {k: v for k, v in r.antecedents.items() if k != "quality"}
                self.rules.append(
                    Rule(r.id, ants, r.consequent, r.description)
                )

        self.output_terms = spec["output_terms"]
        self.fitted_inputs = {
            k: v["source"] for k, v in spec.get("fitted_inputs", {}).items()
        }
        self.fixed_inputs = {
            k: v for k, v in spec.get("fixed_inputs", {}).items()
        }
        self.bounds_: dict = {}
        self.thresholds_: tuple = self.fixed_thresholds
        self.threshold_origin_: str = "not_fitted"
        self._engine: FuzzyEngine | None = None

    # -- fitting ------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "FuzzyStressClassifier":
        X = X.copy()
        input_terms: dict = {}
        self.bounds_ = {}
        for name in self.fitted_inputs:
            col = X[name].to_numpy(float)
            finite = col[np.isfinite(col)]
            if finite.size < 5:
                raise ValueError(f"cannot fit membership bounds for {name}: <5 finite values")
            m1, m2, m3 = np.percentile(finite, [25, 50, 75])
            part = partition_3(None, float(finite.min()), float(m1), float(m2),
                               float(m3), float(finite.max()))
            input_terms[name] = {t: part[t] for t in TERM_LEVELS}
            self.bounds_[name] = part["fit"]
        for name, ispec in self.fixed_inputs.items():
            input_terms[name] = dict(ispec["terms"])
            self.bounds_[name] = "fixed"
        self._engine = FuzzyEngine(input_terms, self.output_terms, self.rules)

        if y is not None and self.threshold_mode == "train_fit":
            scores = [self._engine.infer(dict(row._asdict())).score
                      for row in X.itertuples(index=False)]
            self.thresholds_ = self._fit_thresholds(scores, list(y))
            self.threshold_origin_ = "train_fit"
        else:
            self.thresholds_ = self.fixed_thresholds
            self.threshold_origin_ = "fixed_design_choice"
        return self

    def _fit_thresholds(self, scores, y) -> tuple:
        """Grid-search low/high score cut-offs on TRAINING data only."""
        s = np.array([x if x is not None else np.nan for x in scores], dtype=float)
        valid = np.isfinite(s)
        if valid.sum() < 10 or len(set(np.asarray(y)[valid])) < 2:
            return self.fixed_thresholds
        sv, yv = s[valid], np.asarray(y)[valid]
        best, best_f1 = self.fixed_thresholds, -1.0
        for cut_low in np.arange(0.30, 0.61, 0.05):
            for cut_high in np.arange(max(cut_low + 0.10, 0.40), 0.71, 0.05):
                pred = np.where(sv < cut_low, 0, np.where(sv > cut_high, 1, -1))
                # middle band -> nearest cut-off (needed to compute a 2-class F1)
                mid = pred == -1
                pred[mid] = np.where(
                    sv[mid] - cut_low <= cut_high - sv[mid], 0, 1
                )
                f1 = _macro_f1(yv, pred)
                if f1 > best_f1:
                    best_f1, best = f1, (round(float(cut_low), 2),
                                         round(float(cut_high), 2))
        return best

    # -- prediction ----------------------------------------------------

    @property
    def engine(self) -> FuzzyEngine:
        if self._engine is None:
            raise RuntimeError("FuzzyStressClassifier.fit must be called first")
        return self._engine

    def infer_row(self, row: dict) -> FuzzyOutput:
        out = self.engine.infer(row)
        cut_low, cut_high = self.thresholds_
        if (
            out.score is None
            or out.uncertain_mass >= out.stress_mass
            or out.confidence < self.confidence_tau
        ):
            out.decision = UNCERTAIN
        elif out.score <= cut_low:
            out.decision = "low"
        elif out.score >= cut_high:
            out.decision = "high"
        else:
            out.decision = UNCERTAIN
        return out

    def predict_full(self, X: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in X.itertuples(index=False):
            out = self.infer_row(row._asdict())
            rows.append(
                {
                    "score": out.score,
                    "decision": out.decision,
                    "forced_label": out.forced_label,
                    "confidence": out.confidence,
                    "stress_mass": out.stress_mass,
                    "uncertain_mass": out.uncertain_mass,
                    "n_rules_fired": len(out.fired_rules),
                }
            )
        return pd.DataFrame(rows, index=X.index)

    def predict(self, X: pd.DataFrame):
        return self.predict_full(X)["forced_label"].to_numpy()

    def score_continuous(self, X: pd.DataFrame) -> np.ndarray:
        full = self.predict_full(X)
        return full["score"].fillna(0.5).to_numpy(dtype=float)

    def decision_labels(self, X: pd.DataFrame) -> np.ndarray:
        """Three-way decisions with 'uncertain' for abstention."""
        return self.predict_full(X)["decision"].to_numpy()

    def explain(self, row: dict) -> dict:
        out = self.infer_row(row)
        return {
            "inputs": row,
            "memberships": out.memberships,
            "fired_rules": out.fired_rules,
            "skipped_rules": out.skipped_rules,
            "score": out.score,
            "decision": out.decision,
            "confidence": out.confidence,
            "stress_mass": out.stress_mass,
            "uncertain_mass": out.uncertain_mass,
            "thresholds": list(self.thresholds_),
            "threshold_origin": self.threshold_origin_,
        }


def _macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
