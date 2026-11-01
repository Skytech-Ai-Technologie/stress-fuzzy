# Stress detection evaluation report

Generated: 2026-10-24T16:44:31.471154+00:00  |  protocol: **loso**  |  participants: **20**  |  windows: **696**

> Research prototype for a reproducible experiment on WorkStress3D. Not a medical device, not a diagnostic tool, and not usable for employment decisions. With 20 participants, results below are **exploratory**, not confirmatory.

Class balance (windows): {'0': 459, '1': 237}

## Model comparison (mean ± std over folds)

| model | macro-F1 | balanced acc | stress recall | PR-AUC | coverage |
|---|---|---|---|---|---|
| fuzzy_quality_personalized | 0.481 ± 0.097 | 0.503 ± 0.127 | 0.386 ± 0.224 | 0.430 ± 0.207 | 0.385 ± 0.079 |
| fuzzy_quality_unpersonalized | 0.504 ± 0.097 | 0.530 ± 0.121 | 0.424 ± 0.257 | 0.418 ± 0.208 | 0.538 ± 0.105 |
| fuzzy_no_quality | 0.481 ± 0.097 | 0.503 ± 0.127 | 0.386 ± 0.224 | 0.430 ± 0.207 | 0.419 ± 0.070 |
| logistic_regression | 0.473 ± 0.104 | 0.512 ± 0.132 | 0.531 ± 0.271 | 0.433 ± 0.200 | 1.000 ± 0.000 |
| svm_rbf | 0.553 ± 0.116 | 0.597 ± 0.143 | 0.611 ± 0.264 | 0.507 ± 0.201 | 1.000 ± 0.000 |
| random_forest | 0.634 ± 0.241 | 0.650 ± 0.234 | 0.521 ± 0.351 | 0.615 ± 0.326 | 1.000 ± 0.000 |

Coverage = fraction of windows where the model commits to a label; baselines always commit (1.0). The fuzzy model abstains via its 'uncertain' decision when quality is poor, signals conflict, or confidence is low.

## Coverage vs performance (fuzzy)

| metric | all windows (forced) | decided windows only |
|---|---|---|
| n windows | 696 | 268 |
| macro-F1 | 0.503 | 0.540 |
| balanced acc | 0.504 | 0.539 |

Mean coverage over folds: 0.385 ± 0.079

## Ablations

| variant | macro-F1 | balanced acc | stress recall |
|---|---|---|---|
| fuzzy_quality_personalized | 0.481 ± 0.097 | 0.503 ± 0.127 | 0.386 ± 0.224 |
| fuzzy_quality_unpersonalized | 0.504 ± 0.097 | 0.530 ± 0.121 | 0.424 ± 0.257 |
| fuzzy_no_quality | 0.481 ± 0.097 | 0.503 ± 0.127 | 0.386 ± 0.224 |

## Confusion matrices (all held-out windows pooled)

**fuzzy_quality_personalized** (rows = true non-stress/stress)

| | pred non-stress | pred stress |
|---|---|---|
| true non-stress | 283 | 176 |
| true stress | 144 | 93 |

**fuzzy_quality_unpersonalized** (rows = true non-stress/stress)

| | pred non-stress | pred stress |
|---|---|---|
| true non-stress | 299 | 160 |
| true stress | 129 | 108 |

**fuzzy_no_quality** (rows = true non-stress/stress)

| | pred non-stress | pred stress |
|---|---|---|
| true non-stress | 283 | 176 |
| true stress | 144 | 93 |

**logistic_regression** (rows = true non-stress/stress)

| | pred non-stress | pred stress |
|---|---|---|
| true non-stress | 242 | 217 |
| true stress | 105 | 132 |

**svm_rbf** (rows = true non-stress/stress)

| | pred non-stress | pred stress |
|---|---|---|
| true non-stress | 274 | 185 |
| true stress | 92 | 145 |

**random_forest** (rows = true non-stress/stress)

| | pred non-stress | pred stress |
|---|---|---|
| true non-stress | 367 | 92 |
| true stress | 112 | 125 |

## Participant-level results

Per-participant balanced accuracy is in `artifacts/participant_metrics.csv`; fold-level details in `artifacts/fold_metrics.csv`.

## Statistical comparisons (exploratory)

Wilcoxon signed-rank on per-participant balanced accuracy, fuzzy_quality_personalized vs each other model:

- fuzzy_quality_unpersonalized: n=20, median diff -0.011, p=0.372 (exploratory only; not a confirmatory test)
- fuzzy_no_quality: {'n_pairs': 20, 'test': 'not computable: identical per-participant metrics'}
- logistic_regression: n=20, median diff -0.015, p=0.729 (exploratory only; not a confirmatory test)
- svm_rbf: n=20, median diff -0.065, p=0.011 (exploratory only; not a confirmatory test)
- random_forest: n=20, median diff -0.115, p=0.018 (exploratory only; not a confirmatory test)

## Held-out prediction explanations (examples)

- Subject 1 (true: 0) -> decision **uncertain**, score None, confidence 0.00; fired rules: conflicting_eda_bvp(uncertain, 0.47)
- Subject 1 (true: 1) -> decision **low**, score 0.296, confidence 0.60; fired rules: calm_eda_bvp(low, 0.60); medium_eda_temp(medium, 0.31)
- Subject 1 (true: 1) -> decision **uncertain**, score 0.84, confidence 0.75; fired rules: stress_eda_temp(high, 1.00); conflicting_eda_bvp(uncertain, 1.00)
- Subject 2 (true: 0) -> decision **low**, score 0.166, confidence 0.87; fired rules: calm_eda_bvp(low, 0.87)
- Subject 2 (true: 1) -> decision **uncertain**, score None, confidence 0.00; fired rules: conflicting_eda_bvp(uncertain, 0.41)
- Subject 2 (true: 0) -> decision **uncertain**, score None, confidence 0.00; fired rules: conflicting_eda_bvp(uncertain, 1.00)

Full explanations: `artifacts/explainability_examples.json`.

## Membership bounds

Fuzzy membership limits were estimated from training participants only, per fold: see `artifacts/fuzzy_membership_bounds.json` and `reports/membership_functions.png`.

## Known limitations

- 20 participants; every number here is exploratory.
- Labels come from the dataset's phase-structured protocol (emotion 0/1); the undocumented value 2 was excluded (see audit).
- Audio and facial modalities are blocked: no participant keys, no timestamps, no alignment evidence (see reports/data_audit.md).
- Fuzzy membership values are NOT calibrated probabilities; the score is a rule-based index in [0,1].
- Results are not comparable to other papers' numbers on this or other datasets without identical splits and protocols.
