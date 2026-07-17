"""
Global normalization and dataset splitting.

Implements two splitting strategies:
1. Random split (beat-level) - naive, prone to patient data leakage.
2. Inter-patient split (DS1/DS2, de Chazal et al. 2004 / AAMI standard) -
   the methodologically correct approach.

For each split, one global mean and one global standard deviation are fitted
exclusively on the normal training beats. The same statistics are then
applied to both the training and test sets.

The split is always created before normalization so that no test sample or
test label is used to estimate preprocessing statistics.

Work is skipped only when all expected output files and matching
preprocessing metadata already exist.
"""

import json
import os

import numpy as np
from sklearn.model_selection import train_test_split


# Standard DS1/DS2 split. Paced-rhythm records (102, 104, 107, 217) are
# excluded from both sets per AAMI convention.
DS1_RECORDS = {
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230",
}

DS2_RECORDS = {
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234",
}


def _fit_global_normalizer(X_train):
    """
    Fit one global mean and one global standard deviation using only the
    normal training beats of a split.

    The calculation uses float64 internally for numerical stability. The
    normalized arrays returned by `_apply_global_normalizer` are stored as
    float32 for model training.
    """
    if len(X_train) == 0:
        raise ValueError(
            "Cannot fit global normalization on an empty training set."
        )

    mean = float(np.mean(X_train, dtype=np.float64))
    std = float(np.std(X_train, dtype=np.float64))

    if not np.isfinite(mean):
        raise ValueError("Global normalization mean is not finite.")

    if not np.isfinite(std):
        raise ValueError(
            "Global normalization standard deviation is not finite."
        )

    if std < 1e-8:
        raise ValueError(
            f"Global normalization standard deviation is too small: {std}"
        )

    return mean, std


def _apply_global_normalizer(X, mean, std):
    """
    Apply previously fitted global normalization statistics without
    recalculating them on the target data.
    """
    X_normalized = (X.astype(np.float32) - mean) / std

    if not np.all(np.isfinite(X_normalized)):
        raise ValueError(
            "Global normalization produced non-finite values."
        )

    return X_normalized.astype(np.float32)


def _random_split(X, y, test_size=0.2, random_state=42):
    """
    Create the naive beat-level random split from raw, unnormalized beats.

    Normal beats are divided between training and test sets. All anomalous
    beats are included in the test set.
    """
    X_normal = X[y == 0]
    X_anomalous = X[y == 1]

    X_train, X_test_normal = train_test_split(
        X_normal,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    X_test = np.concatenate(
        [X_test_normal, X_anomalous],
        axis=0,
    )

    y_test = np.concatenate(
        [
            np.zeros(len(X_test_normal), dtype=np.int64),
            np.ones(len(X_anomalous), dtype=np.int64),
        ],
        axis=0,
    )

    return X_train, X_test, y_test


def _inter_patient_split(X, y, record_ids):
    """
    Create the patient-independent DS1/DS2 split from raw beats.

    Training contains only normal beats from DS1 records. Test contains both
    normal and anomalous beats from DS2 records. Paced-rhythm records are
    excluded because they do not belong to either standard subset.
    """
    record_ids = record_ids.astype(str)

    ds1_mask = np.isin(record_ids, list(DS1_RECORDS))
    ds2_mask = np.isin(record_ids, list(DS2_RECORDS))

    X_train = X[ds1_mask & (y == 0)]
    X_test = X[ds2_mask]
    y_test = y[ds2_mask].astype(np.int64)

    return X_train, X_test, y_test


def _metadata_matches(metadata):
    """
    Check whether existing processed arrays were generated with the current
    global-normalization protocol.
    """
    return (
        metadata.get("normalization_strategy") == "global_zscore"
        and metadata.get("fitted_on") == "normal_training_beats_only"
        and metadata.get("split_before_normalization") is True
        and metadata.get("version") == 1
    )


def run(config):
    processed_dir = config["project"]["processed_dir"]

    os.makedirs(processed_dir, exist_ok=True)

    metadata_path = os.path.join(
        processed_dir,
        "preprocessing_metadata.json",
    )

    output_filenames = [
        "X_train_random.npy",
        "X_test_random.npy",
        "y_test_random.npy",
        "X_train_interpatient.npy",
        "X_test_interpatient.npy",
        "y_test_interpatient.npy",
    ]

    outputs_exist = all(
        os.path.exists(os.path.join(processed_dir, filename))
        for filename in output_filenames
    )

    metadata_matches = False

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            metadata_matches = _metadata_matches(metadata)

        except (OSError, json.JSONDecodeError):
            metadata_matches = False

    if outputs_exist and metadata_matches:
        print(
            "[preprocessing] Global-normalized output files already exist, "
            "skipping."
        )
        return

    if outputs_exist and not metadata_matches:
        print(
            "[preprocessing] Existing split files use missing or obsolete "
            "preprocessing metadata; regenerating."
        )

    print("[preprocessing] Loading raw beats...")

    X = np.load(
        os.path.join(processed_dir, "X_beats.npy")
    )
    y = np.load(
        os.path.join(processed_dir, "y_labels.npy")
    )
    record_ids = np.load(
        os.path.join(processed_dir, "record_ids.npy")
    )

    if len(X) != len(y) or len(X) != len(record_ids):
        raise ValueError(
            "X_beats.npy, y_labels.npy, and record_ids.npy have "
            "inconsistent lengths."
        )

    if X.ndim != 2:
        raise ValueError(
            f"Expected beats with shape (n_beats, beat_length), got {X.shape}."
        )

    unique_labels = set(np.unique(y).tolist())

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Expected binary labels 0 and 1, received {unique_labels}."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Raw beat array contains non-finite values."
        )

    print(
        f"[preprocessing] Loaded {len(X):,} beats "
        f"({int(np.sum(y == 0)):,} normal, "
        f"{int(np.sum(y == 1)):,} anomalous)."
    )

    print("[preprocessing] Creating random split on raw beats...")

    X_train_random_raw, X_test_random_raw, y_test_random = (
        _random_split(
            X,
            y,
        )
    )

    print(
        "[preprocessing] Fitting global normalizer on random training set..."
    )

    random_mean, random_std = _fit_global_normalizer(
        X_train_random_raw
    )

    X_train_random = _apply_global_normalizer(
        X_train_random_raw,
        random_mean,
        random_std,
    )

    X_test_random = _apply_global_normalizer(
        X_test_random_raw,
        random_mean,
        random_std,
    )

    print(
        "[preprocessing] Random normalization statistics: "
        f"mean={random_mean:.10f}, std={random_std:.10f}"
    )

    print("[preprocessing] Creating inter-patient split on raw beats...")

    (
        X_train_interpatient_raw,
        X_test_interpatient_raw,
        y_test_interpatient,
    ) = _inter_patient_split(
        X,
        y,
        record_ids,
    )

    print(
        "[preprocessing] Fitting global normalizer on inter-patient "
        "training set..."
    )

    interpatient_mean, interpatient_std = _fit_global_normalizer(
        X_train_interpatient_raw
    )

    X_train_interpatient = _apply_global_normalizer(
        X_train_interpatient_raw,
        interpatient_mean,
        interpatient_std,
    )

    X_test_interpatient = _apply_global_normalizer(
        X_test_interpatient_raw,
        interpatient_mean,
        interpatient_std,
    )

    print(
        "[preprocessing] Inter-patient normalization statistics: "
        f"mean={interpatient_mean:.10f}, "
        f"std={interpatient_std:.10f}"
    )

    np.save(
        os.path.join(processed_dir, "X_train_random.npy"),
        X_train_random,
    )

    np.save(
        os.path.join(processed_dir, "X_test_random.npy"),
        X_test_random,
    )

    np.save(
        os.path.join(processed_dir, "y_test_random.npy"),
        y_test_random,
    )

    np.save(
        os.path.join(processed_dir, "X_train_interpatient.npy"),
        X_train_interpatient,
    )

    np.save(
        os.path.join(processed_dir, "X_test_interpatient.npy"),
        X_test_interpatient,
    )

    np.save(
        os.path.join(processed_dir, "y_test_interpatient.npy"),
        y_test_interpatient,
    )

    metadata = {
        "version": 1,
        "normalization_strategy": "global_zscore",
        "fitted_on": "normal_training_beats_only",
        "split_before_normalization": True,
        "random": {
            "mean": random_mean,
            "std": random_std,
            "training_samples": int(len(X_train_random)),
            "test_samples": int(len(X_test_random)),
            "normal_test_samples": int(
                np.sum(y_test_random == 0)
            ),
            "anomalous_test_samples": int(
                np.sum(y_test_random == 1)
            ),
        },
        "interpatient": {
            "mean": interpatient_mean,
            "std": interpatient_std,
            "training_samples": int(
                len(X_train_interpatient)
            ),
            "test_samples": int(
                len(X_test_interpatient)
            ),
            "normal_test_samples": int(
                np.sum(y_test_interpatient == 0)
            ),
            "anomalous_test_samples": int(
                np.sum(y_test_interpatient == 1)
            ),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"[preprocessing] Saved global-normalized splits to {processed_dir}"
    )

    print(
        f"[preprocessing] Saved metadata to {metadata_path}"
    )