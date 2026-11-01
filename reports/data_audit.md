# WorkStress3D data audit

Generated: 2026-10-24T16:44:00.321711+00:00  |  data dir: `database`

## Verdict
**OK for physiological modelling (multimodal blocked - see alignment)**

| severity | count |
|---|---|
| blocking | 0 |
| warning | 6 |
| info | 7 |

## Findings
- **info**: 15sn: 281 duplicate full rows - consistent with overlapping windowed exports of the same stream, kept as-is
- **warning**: 15sn: label value 2 (undocumented in DataDescripton.pdf) covers 224 rows; assumed ambiguous/transition and excluded from training/evaluation
- **info**: physiological_signals_30sn.csv spells the participant column 'subjet' instead of 'Subject'; normalised to canonical 'subject'
- **info**: 30sn: 594 duplicate full rows - consistent with overlapping windowed exports of the same stream, kept as-is
- **warning**: 30sn: label value 2 (undocumented in DataDescripton.pdf) covers 472 rows; assumed ambiguous/transition and excluded from training/evaluation
- **info**: 60sn: 1209 duplicate full rows - consistent with overlapping windowed exports of the same stream, kept as-is
- **warning**: 60sn: label value 2 (undocumented in DataDescripton.pdf) covers 960 rows; assumed ambiguous/transition and excluded from training/evaluation
- **info**: generalStressTest is a participant-level trait score (TotalPoints); it is NEVER used as a momentary stress label
- **warning**: alignment blocker: facial_expression.csv has no participant column (columns: ['emotion', 'pixels']); images cannot be grouped by participant, so training or evaluating on them would leak identities across folds
- **warning**: alignment blocker: 53 audio sessions found but no session-to-participant mapping file exists; audio cannot be grouped by participant
- **warning**: alignment blocker: instantQuestionnaires.csv has no timestamp column; momentary survey answers cannot be aligned to physiological windows in time
- **info**: alignment note: AudioData/test/ holds 14 wav files with a different naming scheme (e.g. ses_a03.wav, ses_a09.wav, ses_h07.wav); provenance is unclear, so this folder is NOT treated as an independent test set
- **info**: the 15/30/60 s files are windowed exports of the SAME recording sessions (verified: identical first rows and shared phase pattern); only one file is used per experiment to avoid duplicating data

## Files
| key | path | exists |
|---|---|---|
| 15sn | PhysiologicalSignals/physiological_signals_15sn.csv | True |
| 30sn | PhysiologicalSignals/physiological_signals_30sn.csv | True |
| 60sn | PhysiologicalSignals/physiological_signals_60sn.csv | True |
| instant_questionnaires | Survey/instantQuestionnaires.csv | True |
| general_stress_test | Survey/generalStressTest.csv | True |
| panas | Survey/PANAS.csv | True |
| demographic | Survey/demographic.csv | True |
| facial_expression | TheFacialExpressions/facial_expression.csv | True |
| audio_dir | AudioData | True |

Discovered CSV/PDF files: `DataDescripton.pdf`, `PhysiologicalSignals\physiological_signals_15sn.csv`, `PhysiologicalSignals\physiological_signals_30sn.csv`, `PhysiologicalSignals\physiological_signals_60sn.csv`, `Survey\demographic.csv`, `Survey\generalStressTest.csv`, `Survey\instantQuestionnaires.csv`, `Survey\PANAS.csv`, `TheFacialExpressions\facial_expression.csv`

## Physiological files
### 15sn (`PhysiologicalSignals/physiological_signals_15sn.csv`)

- rows: 39200, participants: 20 (column alias `Subject`)
- rows per participant: [1960]
- missing values: none; duplicate rows: 281
- labels: {'0': 25704, '1': 13272, '2': 224} ({'0': 0.6557, '1': 0.3386, '2': 0.0057})
- phases per participant: {'count': 20.0, 'mean': 11.5, 'std': 4.44, 'min': 5.0, '25%': 8.0, '50%': 10.0, '75%': 15.0, 'max': 19.0}
- phase length (samples): {'count': 230.0, 'mean': 170.43, 'std': 200.64, 'min': 56.0, '25%': 56.0, '50%': 112.0, '75%': 168.0, 'max': 1512.0}
- GCD of phase lengths: 56 samples -> natural base window unit
- signal ranges:

| channel | min | max | mean | std | outside plausible |
|---|---|---|---|---|---|
| eda | 0.0 | 24.28 | 1.169 | 2.296 | 0 |
| bvp | -1375.24 | 1280.17 | 0.726 | 121.925 | 0 |
| temp | 24.87 | 39.41 | 33.883 | 2.162 | 0 |
| x | -126.0 | 127.0 | 16.815 | 37.059 | 0 |
| y | -128.0 | 127.0 | 3.305 | 33.246 | 0 |
| z | -128.0 | 127.0 | 23.908 | 31.294 | 0 |

### 30sn (`PhysiologicalSignals/physiological_signals_30sn.csv`)

- rows: 82600, participants: 20 (column alias `subjet`)
- rows per participant: [4130]
- missing values: none; duplicate rows: 594
- labels: {'0': 54162, '1': 27966, '2': 472} ({'0': 0.6557, '1': 0.3386, '2': 0.0057})
- phases per participant: {'count': 20.0, 'mean': 11.5, 'std': 4.44, 'min': 5.0, '25%': 8.0, '50%': 10.0, '75%': 15.0, 'max': 19.0}
- phase length (samples): {'count': 230.0, 'mean': 359.13, 'std': 422.77, 'min': 118.0, '25%': 118.0, '50%': 236.0, '75%': 354.0, 'max': 3186.0}
- GCD of phase lengths: 118 samples -> natural base window unit
- signal ranges:

| channel | min | max | mean | std | outside plausible |
|---|---|---|---|---|---|
| eda | 0.0 | 26.88 | 1.275 | 2.678 | 0 |
| bvp | -2026.95 | 1740.03 | 0.489 | 114.623 | 0 |
| temp | 24.79 | 39.41 | 33.862 | 2.152 | 0 |
| x | -128.0 | 127.0 | 16.722 | 37.088 | 0 |
| y | -128.0 | 127.0 | 4.102 | 33.697 | 0 |
| z | -128.0 | 127.0 | 23.247 | 31.213 | 0 |

### 60sn (`PhysiologicalSignals/physiological_signals_60sn.csv`)

- rows: 168000, participants: 20 (column alias `Subject`)
- rows per participant: [8400]
- missing values: none; duplicate rows: 1209
- labels: {'0': 110160, '1': 56880, '2': 960} ({'0': 0.6557, '1': 0.3386, '2': 0.0057})
- phases per participant: {'count': 20.0, 'mean': 11.5, 'std': 4.44, 'min': 5.0, '25%': 8.0, '50%': 10.0, '75%': 15.0, 'max': 19.0}
- phase length (samples): {'count': 230.0, 'mean': 730.43, 'std': 859.87, 'min': 240.0, '25%': 240.0, '50%': 480.0, '75%': 720.0, 'max': 6480.0}
- GCD of phase lengths: 240 samples -> natural base window unit
- signal ranges:

| channel | min | max | mean | std | outside plausible |
|---|---|---|---|---|---|
| eda | 0.0 | 39.28 | 1.337 | 2.954 | 0 |
| bvp | -2026.95 | 1900.43 | 0.144 | 108.425 | 0 |
| temp | 24.69 | 39.41 | 33.832 | 2.138 | 0 |
| x | -128.0 | 127.0 | 16.336 | 37.415 | 0 |
| y | -128.0 | 127.0 | 4.565 | 34.346 | 0 |
| z | -128.0 | 127.0 | 22.023 | 31.153 | 0 |

## Facial expressions

- rows: 83, labels: {'0': 45, '1': 38}
- participant column: None (none -> cannot be grouped; modality blocked)

## Audio

- session files: 53 (m4a, no participant mapping)
- `test/` wav files (14): ses_a03.wav, ses_a09.wav, ses_h07.wav, ses_n02.wav, ses_n04.wav, ses_n08.wav, ses_n10.wav, ses_n11.wav, ses_n12.wav, ses_sa01.wav, ses_sa05.wav, ses_sa06.wav, ses_sa13.wav, ses_sa14.wav
- different naming scheme from the main sessions; provenance unclear -> NOT an independent test set

## Multimodal alignment

- multimodal allowed: **False**
- blocker: facial_expression.csv has no participant column (columns: ['emotion', 'pixels']); images cannot be grouped by participant, so training or evaluating on them would leak identities across folds
- blocker: 53 audio sessions found but no session-to-participant mapping file exists; audio cannot be grouped by participant
- blocker: instantQuestionnaires.csv has no timestamp column; momentary survey answers cannot be aligned to physiological windows in time
- note: AudioData/test/ holds 14 wav files with a different naming scheme (e.g. ses_a03.wav, ses_a09.wav, ses_h07.wav); provenance is unclear, so this folder is NOT treated as an independent test set

## Assumptions (explicit, reviewable)
- emotion: 0 = non-stress, 1 = stress (per DataDescripton.pdf facial coding; assumed to carry over to the physiological emotion column)
- emotion == 2 is an undocumented, rare (<1%) between-phase marker; treated as ambiguous and excluded from training/evaluation
- rows are 4 Hz samples of one continuous stream per participant (Empatica E4, downsampled per DataDescripton.pdf); labels are constant within protocol phases, not independent per row
- windows are cut inside same-label phases and never span participants
- general stress test / PANAS are participant-level traits, not momentary labels; never used as training targets
- instant questionnaires lack timestamps; not temporally joinable to physiological windows
- audio and facial modalities have no participant key; multimodal fusion is blocked
