# Analysis-of-P300-Brain-Signals-
This is part of a competition that me and 3 of my friends joined, its called International Data Science Challenge. Where built a model that analyses P300 brain signals and display what is a person trying to say. 
# bigP3BCI — P300 Brain-Computer Interface Detection Pipeline

A full end-to-end machine learning pipeline for detecting P300 brainwave responses
from EEG recordings, built for the bigP3BCI competition.

---

## Overview

The P300 is a positive brainwave deflection that occurs approximately 300ms after
a person sees a stimulus they are paying attention to. In a P300 speller BCI system,
rows and columns of characters flash on screen. The row or column containing the
character the user is focusing on will elicit a P300 response. By detecting which
stimulus caused a P300, the system can decode what character the user intended to type.

This pipeline processes raw EEG recordings, extracts features from the P300 time
window, trains and tunes machine learning classifiers, and evaluates their ability
to distinguish target from non-target stimuli.

---

## Dataset

- **Source**: bigP3BCI competition dataset
- **Format**: EDF (European Data Format) files
- **EEG Channels**: 16–32 electrodes per recording (standardised to 16)
- **Sampling Rate**: 256 Hz
- **Epoch Window**: 0–800ms post-stimulus
- **Class Ratio**: ~1:6 (Target : Non-target)

---

## Pipeline Structure

### Phase 1 — Preprocessing
- Bandpass filter: 0.1–30 Hz (retains P300-relevant frequencies)
- Stimulus onset detection using auto-threshold on `StimulusBegin` channel
- Target/non-target label extraction from `StimulusType` channel
- Epoch extraction: 800ms windows starting at each stimulus onset
- Baseline correction: subtract mean of first 100ms from each epoch
- Channel standardisation: keep first 16 EEG channels across all studies
- Batch processing: saves epochs to Google Drive in chunks to manage RAM

### Phase 2 — Feature Engineering
- P300 time window: 250–500ms post-stimulus
- 3 features per channel × 16 channels = **48 features** per epoch:
  - Mean amplitude
  - Peak amplitude
  - Peak latency
- StandardScaler normalisation (fit on train, applied to test)

### Phase 3 — Baseline Model Training
- Logistic Regression (`class_weight='balanced'`)
- Support Vector Machine (`class_weight='balanced'`)
- Random Forest (`class_weight='balanced'`)

### Phase 4 — Hyperparameter Tuning
- **SVM + SMOTE + SelectKBest pipeline** *(recommended best model)*
  - SMOTE oversampling for class imbalance correction
  - SelectKBest feature selection
  - GridSearchCV over C, kernel, and k parameters
- Tuned Logistic Regression (L1/L2 penalty, C sweep)
- Tuned Random Forest (depth, estimators, class weight)

### Phase 5 — Evaluation & Scientific Analysis
- Metrics table: Accuracy, Precision, Recall, F1, AUC
- Confusion matrices for all models
- ROC curves comparison
- Grand average ERP plot at electrode Pz (Target vs Non-target)
- Single-channel (Pz) vs Multi-channel (16ch) performance comparison

### Phase 6 — Advanced Analysis
- **Cross-session generalisation**: Leave-One-Session-Out evaluation
- **Paradigm comparison**: Row-Column (RC) vs Checkerboard (CB)
- **Statistical significance**: Permutation test (100 permutations, p-value)

---

## Model Recommendation

| Model | Strength | Use Case |
|-------|----------|----------|
| **SVM + SMOTE** | Best F1/AUC, robust to imbalance | Primary submission model |
| Logistic Regression | Fast, interpretable, competitive | Explainability / backup |
| Random Forest | Good baseline | Comparison only |

---

## Dependencies

```bash
pip install mne scikit-learn imbalanced-learn matplotlib seaborn pandas numpy
```

| Package | Purpose |
|---------|---------|
| `mne` | EDF file reading and EEG signal processing |
| `scikit-learn` | ML models, GridSearchCV, metrics |
| `imbalanced-learn` | SMOTE oversampling |
| `matplotlib` / `seaborn` | Visualisation |
| `numpy` / `pandas` | Data manipulation |

---

## How to Run

### Setup
1. Upload `bigP3BCI_full_pipeline.py` to **Google Colab**
2. Mount your Google Drive containing the bigP3BCI dataset folder
3. Ensure your data is structured as:
```
MyDrive/
└── bigP3BCI-data/
    ├── StudyA/
    │   └── SubjectID/
    │       └── SessionID/
    │           ├── Train/ParadigmType/*.edf
    │           └── Test/ParadigmType/*.edf
    ├── StudyB/
    └── ...
```

### Configuration
At the top of the file, update these two settings:

```python
# Select which studies to process (or set None for all)
STUDIES = ["StudyA", "StudyB", "StudyC", "StudyD", "StudyO"]

# Set to True only after all chunks are fully processed
RUN_PHASE2 = False
```

### Running Phase 1 (Preprocessing)
The pipeline uses an **auto-run controller** — just keep hitting Run and it
automatically determines which chunk and mode (train/test) to process next.

```
Run 1  → Processes train files 0–200
Run 2  → Processes test files  0–200
Run 3  → Processes train files 200–400
Run 4  → Processes test files  200–400
...    → Continues until all files are done
```

Each run prints its progress:
```
Train chunks done : [0, 200, 400]
Test chunks done  : [0, 200]
Auto-detected: MODE=test | CHUNK_START=200 | CHUNK_END=400
```

When all files are processed:
```
*** ALL FILES PROCESSED! Set RUN_PHASE2 = True to run Phase 2-6 analysis. ***
```

### Running Phase 2–6 (Analysis)
Once all chunks are done, change `RUN_PHASE2 = False` to `RUN_PHASE2 = True`
and run the notebook one final time to execute the full analysis pipeline.

---

## Key Design Decisions

### Auto-threshold for Stimulus Detection
The `StimulusBegin` channel contains analog voltage values rather than clean
digital 0/1 signals due to MNE's EDF scaling. A hardcoded threshold would fail
across files. We compute the midpoint between min and max per file:
```python
threshold = (stimulus_channel.min() + stimulus_channel.max()) / 2
```

### Channel Standardisation
Different studies in the dataset were recorded with 16 or 32 EEG electrodes.
To ensure all epochs can be concatenated, we standardise to the first 16 channels:
```python
eeg_channels = eeg_channels[:16]
```

### Class Imbalance Handling
P300 datasets have a natural ~1:6 Target:Non-target imbalance. We address this
using three complementary strategies:
- `class_weight='balanced'` in all models
- SMOTE synthetic oversampling in the SVM pipeline
- F1-score as the primary evaluation metric (not accuracy)

### Memory Management
With 3000+ EDF files, loading everything at once exceeds Free Colab's ~12GB RAM.
We process files in batches of 10, save to Google Drive, then free RAM:
```python
del raw, eeg_data, epochs
gc.collect()
```

### Drive Disconnection Resilience
Long-running jobs on Free Colab frequently lose their Google Drive connection.
Every file read is wrapped in a retry loop that remounts Drive up to 3 times
before skipping the file.

---

## Output Files

| File | Description |
|------|-------------|
| `confusion_matrices.png` | Confusion matrices for all 3 models |
| `roc_curves.png` | ROC curves comparison |
| `erp_pz.png` | Grand average ERP at electrode Pz |
| `single_vs_multi_channel.png` | Single vs multi-channel performance |
| `paradigm_comparison.png` | RC vs CB paradigm comparison |
| `permutation_test.png` | Statistical significance test |

---

## Authors

Kambing Muda (Sattiya, Sidney, Ika, Ze Dong) 
Built for the bigP3BCI Machine Learning Competition
