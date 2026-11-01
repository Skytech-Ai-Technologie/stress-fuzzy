"""Non-fuzzy baselines: same features, same grouped folds, fold-local fits.

Each estimator is a sklearn Pipeline whose preprocessing (median imputation,
standardisation) is fitted inside ``fit`` - i.e. on the training partition of
each fold only. Participant identifiers are never part of X.
"""
from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVCCalibratedScores(BaseEstimator, ClassifierMixin):
    """SVC with class-weighted decision labels and calibrated score output.

    Replaces the deprecated ``SVC(probability=True)`` while keeping its
    semantics: ``predict`` uses the class-weighted decision function, and
    ``predict_proba`` comes from a CalibratedClassifierCV refitted on the
    SAME training partition only (cv="prefit"), so no held-out data ever
    touches calibration.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, y):
        self._base = SVC(
            kernel="rbf", class_weight="balanced", random_state=self.seed
        )
        self._base.fit(X, y)
        # sklearn >= 1.6 uses FrozenEstimator; older versions use cv="prefit"
        try:
            from sklearn.frozen import FrozenEstimator

            self._cal = CalibratedClassifierCV(
                FrozenEstimator(self._base), ensemble=False
            )
        except ImportError:
            self._cal = CalibratedClassifierCV(
                self._base, ensemble=False, cv="prefit"
            )
        self._cal.fit(X, y)
        self.classes_ = self._base.classes_
        return self

    def predict(self, X):
        return self._base.predict(X)

    def predict_proba(self, X):
        return self._cal.predict_proba(X)


def make_baselines(seed: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", SVCCalibratedScores(seed=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
