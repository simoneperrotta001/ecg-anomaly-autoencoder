"""
Block 2: Preprocessing and train/test split.

Idea: normalize each beat independently (z-score), then split the data so
that the training set contains ONLY normal beats (the autoencoder must learn
what "normal" looks like), while the test set contains both normal and
anomalous beats (needed to evaluate anomaly detection performance).
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split

BASE_DIR = "/storage/internal_02/sperrotta-data/datasets"

def load_data():
    """Load the extracted beats and labels produced by 01_data_prep.py."""
    X = np.load(os.path.join(BASE_DIR, "X_beats.npy"))
    y = np.load(os.path.join(BASE_DIR, "y_labels.npy"))
    return X, y

def normalize_beats(X):
    """
    Z-score normalization per beat (not per dataset): each beat is normalized
    using its own mean and standard deviation. This removes patient-specific
    amplitude differences and keeps the focus on the beat's shape.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1e-8  # avoid division by zero for flat signals
    return (X - mean) / std

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split strategy:
    - Separate normal (label 0) and anomalous (label 1) beats.
    - Training set: only normal beats.
    - Test set: a held-out portion of normal beats + all anomalous beats.
    """
    X_normal = X[y == 0]
    X_anomalous = X[y == 1]

    X_train, X_test_normal = train_test_split(
        X_normal, test_size=test_size, random_state=random_state
    )

    # Test set = held-out normal beats + all anomalous beats
    X_test = np.concatenate([X_test_normal, X_anomalous], axis=0)
    y_test = np.concatenate([
        np.zeros(len(X_test_normal)),
        np.ones(len(X_anomalous))
    ])

    return X_train, X_test, y_test

if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data()
    print(f"Total beats: {X.shape[0]}  (normal={sum(y==0)}, anomalous={sum(y==1)})")

    print("Normalizing beats...")
    X_norm = normalize_beats(X)

    print("Splitting into train/test...")
    X_train, X_test, y_test = split_data(X_norm, y)

    print(f"Train set (normal only): {X_train.shape}")
    print(f"Test set: {X_test.shape}  (normal={sum(y_test==0)}, anomalous={sum(y_test==1)})")

    np.save(os.path.join(BASE_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(BASE_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(BASE_DIR, "y_test.npy"), y_test)
    print(f"Saved train/test splits to {BASE_DIR}")