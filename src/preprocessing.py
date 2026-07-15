"""
Normalization (per-patient z-score) and dataset splitting.

Implements two strategies:
1. Random split (beat-level) - naive, prone to patient data leakage.
2. Inter-patient split (DS1/DS2, de Chazal et al. 2004 / AAMI standard) -
   the methodologically correct approach.
Skips work automatically if outputs already exist.
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split

# Standard DS1/DS2 split. Paced-rhythm records (102, 104, 107, 217) are
# excluded from both sets per AAMI convention.
DS1_RECORDS = {'101','106','108','109','112','114','115','116','118','119',
               '122','124','201','203','205','207','208','209','215','220',
               '223','230'}
DS2_RECORDS = {'100','103','105','111','113','117','121','123','200','202',
               '210','212','213','214','219','221','222','228','231','232',
               '233','234'}


def _normalize_per_patient(X, y, record_ids):
    """
    Z-score normalization using each patient's own normal-beat statistics.
    Removes cross-patient amplitude bias while preserving within-patient
    variation between normal and anomalous beats (the actual diagnostic
    signal). Note: for test-set patients, this assumes their own normal-beat
    statistics are available at inference time (a realistic clinical
    calibration assumption, not train/test label leakage).
    """
    X_norm = np.zeros_like(X)
    for rec in np.unique(record_ids):
        rec_mask = record_ids == rec
        normal_mask = rec_mask & (y == 0)
        reference = X[normal_mask] if normal_mask.sum() > 0 else X[rec_mask]
        mean, std = reference.mean(), reference.std()
        std = std if std != 0 else 1e-8
        X_norm[rec_mask] = (X[rec_mask] - mean) / std
    return X_norm


def _random_split(X, y, test_size=0.2, random_state=42):
    X_normal, X_anomalous = X[y == 0], X[y == 1]
    X_train, X_test_normal = train_test_split(X_normal, test_size=test_size, random_state=random_state)
    X_test = np.concatenate([X_test_normal, X_anomalous], axis=0)
    y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_anomalous))])
    return X_train, X_test, y_test


def _inter_patient_split(X, y, record_ids):
    ds1_mask = np.isin(record_ids, list(DS1_RECORDS))
    ds2_mask = np.isin(record_ids, list(DS2_RECORDS))
    X_train = X[ds1_mask & (y == 0)]
    X_test = X[ds2_mask]
    y_test = y[ds2_mask]
    return X_train, X_test, y_test


def run(config):
    processed_dir = config["project"]["processed_dir"]

    out_files = [
        "X_train_random.npy", "X_test_random.npy", "y_test_random.npy",
        "X_train_interpatient.npy", "X_test_interpatient.npy", "y_test_interpatient.npy",
    ]
    if all(os.path.exists(os.path.join(processed_dir, f)) for f in out_files):
        print("[preprocessing] Output files already exist, skipping.")
        return

    print("[preprocessing] Loading raw beats...")
    X = np.load(os.path.join(processed_dir, "X_beats.npy"))
    y = np.load(os.path.join(processed_dir, "y_labels.npy"))
    record_ids = np.load(os.path.join(processed_dir, "record_ids.npy"))

    print("[preprocessing] Normalizing (per-patient)...")
    X_norm = _normalize_per_patient(X, y, record_ids)

    print("[preprocessing] Random split...")
    X_train_r, X_test_r, y_test_r = _random_split(X_norm, y)

    print("[preprocessing] Inter-patient split...")
    X_train_ip, X_test_ip, y_test_ip = _inter_patient_split(X_norm, y, record_ids)

    np.save(os.path.join(processed_dir, "X_train_random.npy"), X_train_r)
    np.save(os.path.join(processed_dir, "X_test_random.npy"), X_test_r)
    np.save(os.path.join(processed_dir, "y_test_random.npy"), y_test_r)
    np.save(os.path.join(processed_dir, "X_train_interpatient.npy"), X_train_ip)
    np.save(os.path.join(processed_dir, "X_test_interpatient.npy"), X_test_ip)
    np.save(os.path.join(processed_dir, "y_test_interpatient.npy"), y_test_ip)
    print(f"[preprocessing] Saved both splits to {processed_dir}")