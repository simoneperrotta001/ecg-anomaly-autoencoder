"""
Download the MIT-BIH Arrhythmia Database and extract individual heartbeats.
Tracks which record each beat comes from, needed for inter-patient splitting.
Skips work automatically if outputs already exist.
"""

import wfdb
import numpy as np
import os

WINDOW = 180

RECORD_NAMES = [
    '100','101','103','105','106','108','109','111','112','113','114','115','116','117',
    '118','119','121','122','123','124','200','201','202','203','205','207','208','209',
    '210','212','213','214','215','217','219','220','221','222','223','228','230','231',
    '232','233','234'
]


def _record_already_downloaded(data_dir, record_name):
    required_ext = ['.dat', '.hea', '.atr']
    return all(os.path.exists(os.path.join(data_dir, record_name + ext)) for ext in required_ext)


def _download_records(data_dir):
    for rec in RECORD_NAMES:
        if _record_already_downloaded(data_dir, rec):
            continue
        print(f"  Record {rec}: downloading...")
        wfdb.dl_database('mitdb', dl_dir=data_dir, records=[rec])


def _extract_beats(data_dir, record_name):
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, 0]
    beats, labels, record_ids = [], [], []

    NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}

    for peak, symbol in zip(annotation.sample, annotation.symbol):
        start, end = peak - WINDOW, peak + WINDOW
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end]
        label = 0 if symbol in NORMAL_SYMBOLS else 1
        beats.append(beat)
        labels.append(label)
        record_ids.append(record_name)

    return np.array(beats), np.array(labels), np.array(record_ids)


def run(config):
    """Entry point called by main.py. Skips entirely if outputs already exist."""
    processed_dir = config["project"]["processed_dir"]
    data_dir = os.path.join(config["project"]["base_dir"], "mitdb")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    out_files = ["X_beats.npy", "y_labels.npy", "record_ids.npy"]
    if all(os.path.exists(os.path.join(processed_dir, f)) for f in out_files):
        print("[data_prep] Output files already exist, skipping.")
        return

    print("[data_prep] Downloading records...")
    _download_records(data_dir)

    print("[data_prep] Extracting beats...")
    all_beats, all_labels, all_record_ids = [], [], []
    for rec in RECORD_NAMES:
        try:
            beats, labels, record_ids = _extract_beats(data_dir, rec)
            all_beats.append(beats)
            all_labels.append(labels)
            all_record_ids.append(record_ids)
        except Exception as e:
            print(f"  ERROR on record {rec}: {e}")

    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    record_ids = np.concatenate(all_record_ids, axis=0)

    print(f"[data_prep] Final dataset: {X.shape[0]} beats "
          f"(normal={(y==0).sum()}, anomalous={(y==1).sum()})")

    np.save(os.path.join(processed_dir, "X_beats.npy"), X)
    np.save(os.path.join(processed_dir, "y_labels.npy"), y)
    np.save(os.path.join(processed_dir, "record_ids.npy"), record_ids)
    print(f"[data_prep] Saved to {processed_dir}")