# Explainable Fuzzy Stress Detection on WorkStress3D

A reproducible research prototype that audits the WorkStress3D dataset and
evaluates an interpretable, quality-aware fuzzy-logic model for momentary
stress classification from wrist-worn physiological signals, against
transparent unimodal baselines, under **participant-level holdout**
(Leave-One-Subject-Out).

> **This is a research prototype.** It is not a medical device, not a
> diagnostic tool, and must not be used to make employment or other decisions
> about individuals. With 20 participants, all results are **exploratory**.

---

## What it does (and does not do)

- Audits every file in the dataset before any modelling: files, participants,
  labels, ranges, duplicates, alignment, and explicit assumptions.
- Segments the 4 Hz physiological streams into non-overlapping windows inside
  same-label protocol phases (a window never mixes labels or participants).
- Trains a **Mamdani fuzzy inference system** with readable rules
  (`src/stress_fuzzy/rules/fuzzy_rules.yaml`), quality/uncertainty gating,
  per-fold membership bounds fitted on training participants only, and
  per-window explanations (memberships + fired rules).
- Compares against logistic regression, RBF SVM, and random forest with
  identical grouped folds; runs ablations (quality-awareness, personalisation).
- **Blocks multimodal fusion**: the shipped audio and facial files have no
  participant keys and no timestamps, so alignment cannot be demonstrated
  (see `reports/data_audit.md`).

It does **not** claim effectiveness: in the current real-data run, no model
clearly beat chance under LOSO (see Results below). Fuzzy membership values
are rule-based indices, not calibrated probabilities.

## Setup

Python 3.10+ recommended (tested on 3.11, pandas 3.x, scikit-learn 1.9).

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` installs the dependencies and this package in editable
mode, so `python -m stress_fuzzy...` works from the repository root.

### Data placement

Place the extracted WorkStress3D dataset either under `data/WorkStress3D/`
or point the CLI at it with `--data-dir`. On this machine the data lives in
`database/`, so examples below use that. Expected layout:

```
<data-dir>/
├── PhysiologicalSignals/physiological_signals_{15sn,30sn,60sn}.csv
├── Survey/{instantQuestionnaires,generalStressTest,PANAS,demographic}.csv
├── TheFacialExpressions/facial_expression.csv
├── AudioData/            (+ optional test/ subfolder)
└── DataDescripton.pdf
```

## Commands

```bash
# 1. Data audit -> reports/data_audit.md + artifacts/data_audit.json
python -m stress_fuzzy.validation --data-dir database

# 2. Full grouped evaluation + reports
python -m stress_fuzzy.pipeline --data-dir database --protocol loso
#    quicker dev runs: --protocol gk5   (GroupKFold, 5 splits)
#    ablations are run automatically; disable via
#    --no-personalize / --no-quality; custom rules via --rules path.yaml

# 3. Dashboard (data audit, evaluation, explain-a-prediction, export)
streamlit run app.py

# 4. Tests (synthetic fixtures only - never scientific evidence)
python -m pytest tests
```

### Outputs

| path | content |
|---|---|
| `reports/data_audit.md` | human-readable audit + assumptions |
| `artifacts/data_audit.json` | machine-readable audit |
| `reports/evaluation_report.md` | full evaluation report |
| `artifacts/evaluation_results.json` | per-fold, per-participant, aggregate metrics |
| `artifacts/fold_metrics.csv`, `artifacts/participant_metrics.csv` | tabular metrics |
| `artifacts/fuzzy_membership_bounds.json` | membership limits + thresholds, per fold |
| `artifacts/explainability_examples.json` | held-out windows with memberships & fired rules |
| `artifacts/environment.json` | package versions + full config snapshot |
| `reports/*.png` | confusion matrices, membership function plot |

## Label and data assumptions (review before trusting any number)

1. `emotion` 0 = non-stress, 1 = stress, following the dataset's facial
   coding in `DataDescripton.pdf`; assumed to carry over to the
   physiological `emotion` column.
2. `emotion == 2` is undocumented, appears once (~14 s) in 4/20 participants
   between phases (0.57% of rows). Assumed ambiguous/transition and
   **excluded** from training/evaluation.
3. The 15/30/60 s files are windowed exports of the **same sessions**
   (verified: shared phase pattern); only one file is used per experiment.
   The 30 s file spells the participant column `subjet` - normalised to
   `subject`, original preserved and reported.
4. Rows are 4 Hz samples; labels are constant within protocol phases, so
   adjacent rows are NOT independent. Windows are cut inside phases; the 15s
   file's natural base unit (56 samples) is the default window.
5. `generalStressTest` (TotalPoints) and PANAS are participant-level traits -
   never used as momentary labels. `instantQuestionnaires` has no timestamps,
   so it cannot be aligned to physiological windows.
6. `AudioData/test/` wav files use a different naming scheme with unclear
   provenance - NOT treated as an independent test set.

## Evaluation protocol

- **Leave-One-Subject-Out** (primary): each of the 20 participants is held
  out once; no window of the held-out participant enters training or any fit
  step. `GroupKFold(5)` available for quick iteration.
- All preprocessing (imputation, scaling, SVM score calibration), fuzzy
  membership bounds, and decision thresholds are fitted **inside each fold on
  training data only**. No oversampling anywhere; imbalance handled by
  `class_weight="balanced"` on training labels.
- Participant IDs, demographics, and session metadata are never features.
- Leakage is asserted at runtime in every fold (and unit-tested).
- Personalisation (default on) robustly z-scores fuzzy inputs per
  participant using **that participant's own windows only** (no labels) -
  leakage-safe by construction; ablatable via `--no-personalize`.
- Metrics: macro-F1, balanced accuracy, stress-class recall, PR-AUC,
  confusion matrix, per-class precision/recall, and coverage-vs-performance
  for the fuzzy model's `low/uncertain/high` abstention. Accuracy is
  reported but never primary. Wilcoxon comparisons are exploratory.

## The fuzzy model

**Mamdani, not Sugeno**: Mamdani consequents are linguistic terms
(low/medium/high stress), so every fired rule can be read aloud - which is
the point of an interpretable model. Inference: min for AND, max
aggregation, centroid defuzzification over a [0,1] stress universe. The
engine is a small, unit-tested implementation (`fuzzy_model.py`); no
external fuzzy framework is used, so behaviour is fully specified here.

Inputs: `eda_level`, `bvp_variability`, `temp_deviation`, `motion`
(fitted from training quantiles 25/50/75 with saturating shoulder
trapezoids) and `quality` (fixed [0,1] terms). Rules live in
`src/stress_fuzzy/rules/fuzzy_rules.yaml` - including uncertainty rules
(poor quality, motion artifact, conflicting EDA/BVP) that drive an
`uncertain` decision instead of a forced label. Output per window:
score in [0,1], `low/uncertain/high` decision, confidence, fired rules,
and per-input memberships. Decision thresholds are grid-searched on
training scores only (`threshold_mode: train_fit`) or fixed and labelled as
design choices (`fixed`).

## Results (real data, LOSO, 20 participants, 696 windows)

Class balance 459 non-stress / 237 stress. **Exploratory** - no model
clearly beats chance; differences between models are within fold-to-fold
variability:

| model | macro-F1 | balanced acc | stress recall | coverage |
|---|---|---|---|---|
| fuzzy (quality+personalized) | 0.481 ± 0.097 | 0.503 ± 0.127 | 0.386 | 0.38 |
| fuzzy (quality, unpersonalized) | 0.504 ± 0.097 | 0.530 ± 0.121 | 0.424 | 0.54 |
| fuzzy (no quality gating) | 0.481 ± 0.097 | 0.503 ± 0.127 | 0.386 | 0.42 |
| logistic regression | 0.473 ± 0.104 | 0.512 ± 0.132 | 0.531 | 1.00 |
| SVM (RBF) | 0.553 ± 0.116 | 0.597 ± 0.143 | 0.611 | 1.00 |
| random forest | 0.634 ± 0.241 | 0.650 ± 0.234 | 0.521 | 1.00 |

Honest reading: under participant-level holdout, physiological windows from
this dataset do not support confident stress detection with these features.
The fuzzy model abstains often (38% coverage) - its covered-only metrics are
in `reports/evaluation_report.md`. Personalisation did not help here.
These numbers are NOT comparable to results reported in other papers on this
dataset without identical splits and protocols.

## Contribution pilots (exploratory)

`pilots.py` probes three research directions on the real data; outputs go to
`bench/pilots/` (see `pilot_results.md`). Findings (all exploratory, n=20):

- **Personal few-shot calibration does not help.** Adding up to 20 labeled
  windows of the held-out person to training (BA flat at ~0.60), or
  re-fitting only the decision threshold on them (~0.59-0.61), never lifts
  LOSO performance - personalization is not the bottleneck.
- **Evaluation-protocol inflation is small here** (random-window CV vs LOSO:
  only +0.02-0.04 balanced accuracy) because models cannot even exploit
  leaked subject structure; in-sample fit is high (RF 0.99), so features
  mostly encode person-specific quirks, not transferable stress signal.
- **Persistence-prior Viterbi smoothing hurts** (0.582 -> 0.550) - temporal
  smoothing of scores misaligns with the phase structure.

Remaining evidence-backed directions: feature-level signal processing (EDA
tonic/phasic decomposition) and/or publishing the audit + grouped benchmark
as a rigorous negative result.

## WESAD benchmark (`wesad_benchmark.py`)

The same quality-aware fuzzy machinery was ported to WESAD (Schmidt et al.
2018; 15 subjects, Empatica E4 wrist + RespiBAN chest, synchronized views,
real stress labels) to test the methods on a dataset where a transferable
stress signal actually exists. Data via `kagglehub` (UCI pointer zip links a
dead mirror). Non-overlapping 60 s windows in contiguous label phases, LOSO
by subject, fold-local fitting, inner-GroupKFold out-of-fold scores for the
fusion layer (leakage-safe). HRV features from 64 Hz BVP / 700 Hz ECG.

Binary task (stress vs non-stress), 547 windows, balanced accuracy:

| model | balanced acc | stress recall |
|---|---|---|
| fuzzy direct (hand rules, no HRV) | 0.560 | 0.418 |
| fuzzy direct + HRV rules | 0.582 | 0.418 |
| logistic regression | 0.777 | 0.788 |
| SVM wrist-only / chest-only | 0.862 / 0.839 | 0.824 / 0.745 |
| random forest | 0.881 | 0.848 |
| **fuzzy multimodal fusion (Mamdani over calibrated wrist+chest scores)** | **0.879** | 0.818 |
| fuzzy fusion, covered windows only (83% coverage) | **0.905** | 0.836 |

Reading: the interpretable fusion layer matches the best black-box baseline
(0.879 vs 0.881) while adding abstention (conflict/quality rules flag 17% of
windows as uncertain) and per-window rule explanations; where it commits it
reaches 0.905 balanced accuracy. Hand-crafted direct fuzzy rules top out at
~0.58 - the fusion of learned evidence with an interpretable decision layer,
not hand rules alone, is what makes fuzzy competitive here. 3-class
(baseline/stress/amusement): RF 0.641, in line with published wrist results.
All numbers exploratory (15 subjects).

## Cross-dataset transfer (`cross_dataset_transfer.py`)

Both datasets come from the same Empatica E4 wrist, so a harmonized feature
space of identical-semantics channels (EDA mean/std/slope at 4 Hz, TEMP
mean/std, ACC-magnitude motion) allows a direct transfer test. No
target-domain data is used for fitting; in-dataset LOSO controls use the
same features.

| direction | balanced accuracy (best model) |
|---|---|
| control: WESAD in-dataset LOSO | 0.835 (RF) |
| control: WorkStress3D in-dataset LOSO | 0.660 (RF) |
| transfer WESAD -> WorkStress3D | 0.510 (LR) — chance |
| transfer WorkStress3D -> WESAD | 0.505 (LR) — chance |

Transfer collapses to chance in both directions; the WorkStress3D-trained
models call 82% of WESAD windows "stress", the WESAD-trained models default
toward non-stress on WorkStress3D, and no WorkStress3D participant reaches
0.7 under the WESAD-trained model. Conclusion: the harmonized 4 Hz wrist
space carries no dataset-invariant stress signature — physiological stress
detectors are protocol-bound (lab TSST vs field phases), which is the
mechanistic explanation for both the WorkStress3D negative result and the
necessity of per-dataset evaluation. Together with the leakage ladder this
completes the multi-dataset program: signal exists (WESAD), is learnable
(fusion 0.879/0.905 covered), does not transfer across protocols, and high
published numbers on field data deserve leakage scrutiny.

## Privacy

All processing is local; nothing is sent to external services. `data/`,
`database/`, `artifacts/`, and raw media are git-ignored - never commit the
dataset. The dashboard shows only anonymised participant numbers and never
plays audio or shows face images.

## Known limitations

- 20 participants; every metric is exploratory with wide uncertainty.
- Labels derive from a phase-structured protocol, not per-moment ground
  truth; label provenance for `emotion` is itself an assumption.
- Multimodal fusion blocked pending participant keys/timestamps for audio
  and facial data; a non-fuzzy fusion baseline is therefore also out of
  scope until alignment exists.
- The fuzzy score is not a calibrated probability; calibration was not
  measured.

## Project layout

```
├── app.py                      # Streamlit dashboard
├── pyproject.toml              # editable install
├── requirements.txt
├── src/stress_fuzzy/
│   ├── config.py               # all knobs + documented assumptions
│   ├── data.py                 # loading, aliases, windowing, alignment gates
│   ├── validation.py           # audit CLI (python -m stress_fuzzy.validation)
│   ├── features.py             # window stats + quality flags
│   ├── fuzzy_model.py          # Mamdani engine + classifier + explanations
│   ├── rules/fuzzy_rules.yaml  # human-readable rule base (edit me)
│   ├── baselines.py            # LR / SVM / RF pipelines
│   ├── evaluation.py           # LOSO + GroupKFold, leakage guards, metrics
│   ├── reporting.py            # JSON/CSV/Markdown exports
│   └── pipeline.py             # end-to-end CLI (python -m stress_fuzzy.pipeline)
├── tests/                      # synthetic fixtures only
├── data/ or database/          # user-provided dataset (never committed)
├── artifacts/                  # generated (never committed)
└── reports/                    # generated reports
```
