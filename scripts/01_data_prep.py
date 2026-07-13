"""
Block 1: Download the MIT-BIH Arrhythmia Database and extract individual heartbeats.

Idea: each record contains a continuous ECG signal + annotations with the position
of every heartbeat (R-peak) and its label (N = normal, other letters = anomalies).
We cut a fixed window around each R-peak -> we obtain an isolated "beat".
"""

import wfdb
import numpy as np
import os

# Base path: datasets live outside the project folder, in shared external storage
BASE_DIR = "/storage/internal_02/sperrotta-data/datasets"
DATA_DIR = os.path.join(BASE_DIR, "mitdb")   # folder where wfdb downloads records
WINDOW = 180                                  # samples before/after R-peak (total window ~360 samples)
os.makedirs(DATA_DIR, exist_ok=True)

# Standard list of MIT-BIH records (44 recordings)
RECORD_NAMES = [
    '100','101','103','105','106','108','109','111','112','113','114','115','116','117',
    '118','119','121','122','123','124','200','201','202','203','205','207','208','209',
    '210','212','213','214','215','217','219','220','221','222','223','228','230','231',
    '232','233','234'
]

def record_already_downloaded(record_name):
    """Check if the .dat, .hea and .atr files for a record already exist locally."""
    required_ext = ['.dat', '.hea', '.atr']
    return all(
        os.path.exists(os.path.join(DATA_DIR, record_name + ext))
        for ext in required_ext
    )

def download_records():
    """Download signal and annotations for each record, skipping ones already present."""
    for rec in RECORD_NAMES:
        if record_already_downloaded(rec):
            print(f"Record {rec}: already downloaded, skipping.")
            continue
        print(f"Record {rec}: downloading...")
        wfdb.dl_database('mitdb', dl_dir=DATA_DIR, records=[rec])

def extract_beats(record_name):
    """
    Extract individual beats from a record:
    - reads the raw signal (channel 0, typically MLII)
    - reads the annotations (position + type of each beat)
    - crops a fixed window around each R-peak
    Returns: numpy array of beats, numpy array of labels (0=normal, 1=anomalous)
    """
    record_path = os.path.join(DATA_DIR, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, 0]  # first channel
    beats, labels = [], []

    # 'N' = normal beat per AAMI standard; everything else is treated as anomalous
    NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}

    for peak, symbol in zip(annotation.sample, annotation.symbol):
        start, end = peak - WINDOW, peak + WINDOW
        if start < 0 or end > len(signal):
            continue  # discard beats too close to signal boundaries
        beat = signal[start:end]
        label = 0 if symbol in NORMAL_SYMBOLS else 1
        beats.append(beat)
        labels.append(label)

    return np.array(beats), np.array(labels)

def build_dataset():
    """Extract beats from all records and merge them into a single dataset."""
    all_beats, all_labels = [], []
    failed_records = []

    for rec in RECORD_NAMES:
        try:
            beats, labels = extract_beats(rec)
            all_beats.append(beats)
            all_labels.append(labels)
            print(f"Record {rec}: {len(beats)} beats extracted")
        except Exception as e:
            print(f"ERROR on record {rec}: {e}")
            failed_records.append(rec)

    if failed_records:
        print(f"\nWARNING: {len(failed_records)} record(s) failed: {failed_records}")

    if not all_beats:
        raise RuntimeError("No beats extracted from any record. Check errors above.")

    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y

if __name__ == "__main__":
    print(f"Base directory: {BASE_DIR}")
    print(f"Data directory: {DATA_DIR}\n")

    print("=== Step 1: Download ===")
    download_records()

    print("\n=== Step 2: Extract beats ===")
    X, y = build_dataset()

    print(f"\n=== Step 3: Save ===")
    print(f"Final dataset: {X.shape[0]} beats, {X.shape[1]} samples each")
    print(f"Normal: {(y==0).sum()}  |  Anomalous: {(y==1).sum()}")

    x_path = os.path.join(BASE_DIR, "X_beats.npy")
    y_path = os.path.join(BASE_DIR, "y_labels.npy")
    np.save(x_path, X)
    np.save(y_path, y)

    print(f"Saved: {x_path}")
    print(f"Saved: {y_path}")
    print("\nDone.")