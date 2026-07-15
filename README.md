# ECG Anomaly Detection with a 1D Convolutional Autoencoder

Project for the Machine Learning course exam — Università Parthenope.

**Topic:** Autoencoders
**Scope:** Biomedical Signal Processing

## Overview

This project implements an unsupervised anomaly detection pipeline for ECG
heartbeats. Autoencoders are trained **only on normal heartbeats**. At
inference time, the model is expected to reconstruct normal beats
accurately, while reconstructing anomalous beats (arrhythmias) poorly. The
reconstruction error is therefore used as an anomaly score.

The project also includes two experimental analyses within the same
Autoencoders topic: a comparison between convolutional (Conv1D) and
fully-connected (Dense) encoder/decoder designs, and a sensitivity study on
the autoencoder's bottleneck width.

## Why an autoencoder for this task

Labeling every possible arrhythmia type is expensive and requires expert
cardiologist annotation for every class. By training only on the majority
class (normal beats), the model learns a compact representation of what a
"normal" heartbeat looks like, without needing balanced or fully labeled data
for every anomaly type. This is a standard approach in biomedical anomaly
detection literature.

## Core methodological contribution: random vs. inter-patient splitting

Beyond implementing the standard autoencoder-based approach, this project
explicitly investigates a common pitfall in beat-level ECG classification:
**patient data leakage**. Two splitting strategies are implemented and
compared using the exact same model and training procedure:

1. **Random split (naive):** beats are split randomly regardless of which
   patient they come from. Beats from the same patient can appear in both
   train and test sets, inflating performance metrics.
2. **Inter-patient split (correct):** based on the standard DS1/DS2 split
   defined by de Chazal et al. (2004) / AAMI convention. Entire patients are
   held out for testing; no patient appears in both sets. Paced-rhythm
   records (102, 104, 107, 217) are excluded from both sets, per AAMI
   convention.

The random split produces a noticeably higher (optimistic) AUC than the
inter-patient split, demonstrating quantitatively how naive splitting
overstates real-world generalization performance.

## Why a convolutional architecture (Conv1D) instead of a Dense one

An ECG beat is a temporal signal with a characteristic local morphology (the
QRS complex, P wave, T wave). 1D convolutions are designed to capture local,
repeating patterns along a sequence via weight sharing, which makes them
more robust to small temporal shifts (jitter) in R-peak alignment and to
morphological variability between patients, compared to a fully-connected
(Dense) autoencoder that treats every sample as an independent feature. This
project verifies that claim empirically (see "Architecture comparison"
below) rather than assuming it.

## Dataset

**MIT-BIH Arrhythmia Database** (PhysioNet), 44 half-hour ECG recordings
sampled at 360 Hz, with beat-by-beat annotations made by cardiologists.

- Source: https://physionet.org/content/mitdb/
- Beats are extracted as fixed-length windows (360 samples, 180 before and
  180 after each annotated R-peak).
- Labels follow the AAMI standard: beats annotated as `N`, `L`, `R`, `e`, `j`
  are treated as **normal**; all other annotation symbols are treated as
  **anomalous**.
- Each beat is z-score normalized using its own patient's normal-beat
  statistics (not per-beat), which removes cross-patient amplitude bias
  while preserving within-patient variation between normal and anomalous
  beats — the actual diagnostic signal.
- Data lives entirely outside the project folder, in shared external
  storage, and is not versioned in this repository (see `.gitignore`). It
  is downloaded and regenerated automatically the first time the pipeline
  runs.

## Project structure

    ecg_autoencoder_project/
    ├── configs/
    │   └── experiments.yaml     # single source of truth: every model run to train
    ├── src/
    │   ├── __init__.py
    │   ├── data_prep.py         # download MIT-BIH, extract beats
    │   ├── preprocessing.py     # per-patient normalization, random + inter-patient splits
    │   ├── engine.py            # shared train/evaluate logic, used by every run
    │   └── models/
    │       ├── __init__.py
    │       ├── conv_autoencoder.py   # Conv1D autoencoder, configurable bottleneck
    │       └── dense_autoencoder.py  # Dense autoencoder baseline
    ├── main.py                   # SINGLE entry point: runs the entire study end-to-end
    ├── scripts/
    │   ├── setup_env.sh          # creates venv, installs requirements.txt
    │   ├── run_all.sh            # direct execution (no scheduler)
    │   └── run_all.sbatch        # SLURM execution (if a scheduler is available)
    ├── models/                    # trained checkpoints, one per run id (not versioned)
    ├── results/                    # plots, metrics, evaluation outputs (not versioned)
    ├── requirements.txt
    ├── README.md
    └── .gitignore

Note: datasets are not stored inside this repository. They live in external
shared storage and are downloaded/regenerated automatically by `main.py`.

## How to reproduce

1. Create the virtual environment and install dependencies:
```bash
   bash scripts/setup_env.sh
   source venv/bin/activate
```

2. Run the entire study (data download, preprocessing, all training runs,
   all evaluations, and comparison reports) in a single command:
```bash
   bash scripts/run_all.sh
```
   or, if a SLURM scheduler is available on the execution machine:
```bash
   sbatch scripts/run_all.sbatch
```

   `main.py` automatically skips any step whose output already exists
   (downloaded records, processed splits, or a trained checkpoint), so
   re-running it after an interruption resumes from where it left off
   instead of redoing completed work.

## Experiment configuration (`configs/experiments.yaml`)

Every unique model configuration (architecture, split, bottleneck width) is
declared exactly once under `runs`, and trained exactly once. Comparisons
(architecture, bottleneck sensitivity) are reporting-only views over those
same runs — no configuration is ever duplicated or retrained:

- `conv_random` — Conv1D, random split
- `conv_interpatient` — Conv1D, inter-patient split (primary model)
- `dense_interpatient` — Dense baseline, inter-patient split
- `conv_bottleneck_16 / 32 / 128` — Conv1D, inter-patient split, varying
  bottleneck width (64 is covered by `conv_interpatient`, reused rather than
  retrained)

## Model architecture (summary)

**Conv1D autoencoder** — symmetric 1D convolutional encoder-decoder:
- **Encoder:** stacked `Conv1d` layers with stride-2 downsampling,
  progressively compressing 360 → 180 → 90 → 45 samples while increasing
  channel depth toward the configured bottleneck width.
- **Decoder:** stacked `ConvTranspose1d` layers mirroring the encoder,
  upsampling back to 360 samples.
- **Loss:** Mean Squared Error (MSE) between input and reconstruction.

**Dense autoencoder baseline** — fully-connected encoder-decoder with the
same input/output dimensionality (360) and a matched latent dimension, used
only as an architectural comparison point.

Both are trained exclusively on normal beats.

## Evaluation methodology

- The reconstruction error (MSE per beat) is computed on the held-out test
  set (normal + anomalous beats).
- Two threshold strategies are computed and compared for every run:
  - **Youden's J statistic** (maximizes TPR − FPR): the primary operating
    point used, since in a clinical screening context missing a true
    arrhythmia (false negative) is generally more costly than a false
    alarm that a cardiologist can quickly dismiss.
  - **F1-optimal threshold** (from the Precision-Recall curve): computed
    for comparison, to verify the Youden's J choice is not costing
    meaningful F1 performance (in practice, the gap between the two was
    found to be small, ~0.005–0.02 F1).
- Metrics reported per run: AUC, precision, recall, F1, confusion matrix,
  ROC curve, and Precision-Recall curve.
- All results are consolidated into `results/all_results.json`.

## Notes on dataset imbalance

The MIT-BIH dataset is naturally imbalanced (most beats are normal). This is
not a limitation for this approach, since the autoencoder is trained
exclusively on the normal class; the imbalance is only relevant at
evaluation time, where imbalance-aware metrics (ROC/AUC, precision/recall,
F1) are used instead of raw accuracy.

## Known limitations

- Precision on the inter-patient split is moderate (models trade some false
  positives for higher recall, a deliberate and justified choice given the
  clinical cost asymmetry described above).
- Per-patient normalization assumes that a patient's own normal-beat
  statistics are available at evaluation time (a realistic clinical
  calibration assumption — e.g. a short baseline recording for a new
  patient — rather than train/test label leakage).

## References

- de Chazal P, O'Dwyer M, Reilly RB. Automatic Classification of Heartbeats
  Using ECG Morphology and Heartbeat Interval Features. IEEE Trans Biomed
  Eng 51(7):1196-1206 (2004).
- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
  IEEE Eng in Med and Biol 20(3):45-50 (2001).
- PhysioNet: Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet.
  Circulation 101(23):e215-e220 (2000).