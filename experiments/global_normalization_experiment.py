"""
Global normalization experiment.

This experiment evaluates whether global z-score normalization can satisfy
both of the following conditions:

1. Preserve a valid anomaly-detection score:
   reconstruction error should rank anomalous beats above normal beats,
   resulting in AUC > 0.5.

2. Preserve enough patient-specific information for the performance gap
   between the naive random split and the inter-patient split to remain
   measurable.

The experiment is intentionally isolated from the main pipeline:

- it does not modify the processed split files;
- it does not modify the main checkpoints;
- it does not modify results/all_results.json;
- it does not use per-patient or per-beat normalization;
- normalization statistics are fitted only on the normal training beats
  of each split;
- the split is created before fitting the normalization statistics.

The final output is written to:

    results/global_normalization_experiment.json

Youden's J threshold is computed on the test set only as an exploratory
oracle analysis. AUC is the primary threshold-independent metric.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.engine as engine
from src.models.conv_autoencoder import ConvAutoencoder


# ---------------------------------------------------------------------------
# Dataset split definitions
# ---------------------------------------------------------------------------

# Standard DS1/DS2 split from de Chazal et al.
# Paced-rhythm records 102, 104, 107, and 217 are excluded.
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

SEED = 42
BOTTLENECK_CHANNELS = 64
EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """
    Configure deterministic random number generation.

    The seed is reset before each model training so that the random and
    inter-patient experiments start from the same model initialization and
    deterministic training configuration.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def create_random_split(
    X: np.ndarray,
    y: np.ndarray,
    record_ids: np.ndarray,
    test_size: float = 0.2,
    random_state: int = SEED,
):
    """
    Create the naive beat-level random split using raw, unnormalized beats.

    Only normal beats are divided between training and test sets.
    All anomalous beats are placed in the test set.

    Record identifiers are returned only for diagnostic reporting.
    """
    normal_indices = np.flatnonzero(y == 0)
    anomalous_indices = np.flatnonzero(y == 1)

    train_indices, test_normal_indices = train_test_split(
        normal_indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    test_indices = np.concatenate(
        [test_normal_indices, anomalous_indices],
        axis=0,
    )

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_test = y[test_indices].astype(np.int64)

    train_record_ids = record_ids[train_indices]
    test_record_ids = record_ids[test_indices]

    return (
        X_train,
        X_test,
        y_test,
        train_record_ids,
        test_record_ids,
    )


def create_interpatient_split(
    X: np.ndarray,
    y: np.ndarray,
    record_ids: np.ndarray,
):
    """
    Create the patient-independent DS1/DS2 split using raw beats.

    Training:
        normal beats from DS1 records only.

    Test:
        normal and anomalous beats from DS2 records.

    No DS2 sample or label is used to fit normalization statistics.
    """
    record_ids_str = record_ids.astype(str)

    ds1_mask = np.isin(record_ids_str, list(DS1_RECORDS))
    ds2_mask = np.isin(record_ids_str, list(DS2_RECORDS))

    train_mask = ds1_mask & (y == 0)
    test_mask = ds2_mask

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_test = y[test_mask].astype(np.int64)

    train_record_ids = record_ids_str[train_mask]
    test_record_ids = record_ids_str[test_mask]

    return (
        X_train,
        X_test,
        y_test,
        train_record_ids,
        test_record_ids,
    )


# ---------------------------------------------------------------------------
# Global normalization
# ---------------------------------------------------------------------------

def fit_global_normalizer(X_train: np.ndarray):
    """
    Fit one global mean and one global standard deviation.

    Statistics are calculated exclusively from normal training beats.
    No test sample and no test label is used.
    """
    mean = float(np.mean(X_train, dtype=np.float64))
    std = float(np.std(X_train, dtype=np.float64))

    if not np.isfinite(mean):
        raise ValueError("The global training mean is not finite.")

    if not np.isfinite(std):
        raise ValueError("The global training standard deviation is not finite.")

    if std < EPSILON:
        raise ValueError(
            f"The global training standard deviation is too small: {std}"
        )

    return mean, std


def apply_global_normalizer(
    X: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """
    Apply previously fitted global statistics without refitting them.
    """
    X_normalized = (X.astype(np.float32) - mean) / std
    return X_normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def summarize_values(values: np.ndarray) -> dict:
    """
    Return compact descriptive statistics for a one-dimensional array.
    """
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "q25": None,
            "q75": None,
        }

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def count_class_samples(y_test: np.ndarray) -> dict:
    """
    Count normal and anomalous test samples.
    """
    normal_count = int(np.sum(y_test == 0))
    anomalous_count = int(np.sum(y_test == 1))
    total_count = int(len(y_test))

    prevalence = (
        float(anomalous_count / total_count)
        if total_count > 0
        else None
    )

    return {
        "total": total_count,
        "normal": normal_count,
        "anomalous": anomalous_count,
        "anomaly_prevalence": prevalence,
    }


def count_records(record_ids: np.ndarray) -> dict:
    """
    Return the number and sorted identifiers of unique records.
    """
    unique_records = sorted(
        {str(record_id) for record_id in record_ids}
    )

    return {
        "count": len(unique_records),
        "record_ids": unique_records,
    }


def verify_split(
    split_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Validate the most important assumptions before training.
    """
    if X_train.ndim != 2:
        raise ValueError(
            f"{split_name}: expected two-dimensional X_train, "
            f"received shape {X_train.shape}."
        )

    if X_test.ndim != 2:
        raise ValueError(
            f"{split_name}: expected two-dimensional X_test, "
            f"received shape {X_test.shape}."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"{split_name}: train/test beat lengths differ: "
            f"{X_train.shape[1]} and {X_test.shape[1]}."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"{split_name}: X_test and y_test lengths differ."
        )

    unique_labels = set(np.unique(y_test).tolist())

    if unique_labels != {0, 1}:
        raise ValueError(
            f"{split_name}: expected test labels {{0, 1}}, "
            f"received {unique_labels}."
        )

    if not np.all(np.isfinite(X_train)):
        raise ValueError(f"{split_name}: X_train contains non-finite values.")

    if not np.all(np.isfinite(X_test)):
        raise ValueError(f"{split_name}: X_test contains non-finite values.")


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    split_name: str,
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    train_record_ids: np.ndarray,
    test_record_ids: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict:
    """
    Fit global normalization, train one Conv1D autoencoder, and evaluate it.
    """
    verify_split(
        split_name=split_name,
        X_train=X_train_raw,
        X_test=X_test_raw,
        y_test=y_test,
    )

    normalization_mean, normalization_std = fit_global_normalizer(
        X_train_raw
    )

    X_train = apply_global_normalizer(
        X_train_raw,
        normalization_mean,
        normalization_std,
    )

    X_test = apply_global_normalizer(
        X_test_raw,
        normalization_mean,
        normalization_std,
    )

    X_train_tensor = torch.from_numpy(
        X_train.reshape(-1, 1, X_train.shape[1])
    )

    X_test_tensor = torch.from_numpy(
        X_test.reshape(-1, 1, X_test.shape[1])
    )

    # Reset the seed before every model so the comparison uses the same
    # deterministic model initialization and training sequence.
    set_seed(SEED)

    model = ConvAutoencoder(
        bottleneck_channels=BOTTLENECK_CHANNELS
    )

    print()
    print("=" * 72)
    print(f"Training split: {split_name}")
    print("=" * 72)
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples:     {len(X_test):,}")
    print(f"Global mean:      {normalization_mean:.10f}")
    print(f"Global std:       {normalization_std:.10f}")
    print()

    model, loss_history = engine.train(
        model=model,
        X_train=X_train_tensor,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
    )

    errors = engine.compute_errors(
        model=model,
        X_test=X_test_tensor,
        device=device,
    )

    # Existing project evaluation: AUC and oracle Youden threshold.
    metrics = engine.evaluate(
        errors=errors,
        y_test=y_test,
    )

    # Diagnostic reversed-score AUC.
    # This is not used as the final anomaly score; it only quantifies whether
    # the reconstruction-error ranking is informative in the opposite
    # direction.
    reversed_auc = roc_auc_score(
        y_test,
        -errors,
    )

    normal_errors = errors[y_test == 0]
    anomalous_errors = errors[y_test == 1]

    patient_overlap = sorted(
        set(map(str, train_record_ids))
        .intersection(set(map(str, test_record_ids)))
    )

    result = {
        "split": split_name,
        "architecture": "conv1d",
        "bottleneck_channels": BOTTLENECK_CHANNELS,
        "seed": SEED,
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "initial_loss": float(loss_history[0]),
            "final_loss": float(loss_history[-1]),
        },
        "normalization": {
            "strategy": "global_zscore",
            "fitted_on": "normal_training_beats_only",
            "mean": normalization_mean,
            "std": normalization_std,
        },
        "dataset": {
            "training_samples": int(len(X_train)),
            "test_samples": count_class_samples(y_test),
            "training_records": count_records(train_record_ids),
            "test_records": count_records(test_record_ids),
            "shared_train_test_records": {
                "count": len(patient_overlap),
                "record_ids": patient_overlap,
            },
        },
        "metrics": metrics,
        "diagnostics": {
            "auc_forward_error": float(metrics["auc"]),
            "auc_reversed_error": float(reversed_auc),
            "expected_direction_valid": bool(metrics["auc"] > 0.5),
            "error_statistics": {
                "normal": summarize_values(normal_errors),
                "anomalous": summarize_values(anomalous_errors),
            },
            "mean_error_difference_anomalous_minus_normal": float(
                np.mean(anomalous_errors) - np.mean(normal_errors)
            ),
            "median_error_difference_anomalous_minus_normal": float(
                np.median(anomalous_errors) - np.median(normal_errors)
            ),
        },
        "evaluation_note": (
            "The Youden threshold and the associated precision, recall, F1, "
            "and confusion matrix are exploratory oracle metrics because the "
            "threshold is selected using test labels. AUC is the primary "
            "threshold-independent metric."
        ),
    }

    print()
    print(f"Result for {split_name}")
    print(f"  Forward AUC:  {metrics['auc']:.4f}")
    print(f"  Reversed AUC: {reversed_auc:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1:           {metrics['f1']:.4f}")
    print(
        "  Mean error difference "
        f"(anomalous - normal): "
        f"{result['diagnostics']['mean_error_difference_anomalous_minus_normal']:.8f}"
    )

    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "experiments.yaml"
    output_path = (
        PROJECT_ROOT
        / "results"
        / "global_normalization_experiment.json"
    )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    processed_dir = Path(config["project"]["processed_dir"])

    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    learning_rate = float(config["training"]["learning_rate"])

    required_files = {
        "beats": processed_dir / "X_beats.npy",
        "labels": processed_dir / "y_labels.npy",
        "record_ids": processed_dir / "record_ids.npy",
    }

    missing_files = [
        str(path)
        for path in required_files.values()
        if not path.exists()
    ]

    if missing_files:
        formatted_missing = "\n".join(
            f"  - {path}" for path in missing_files
        )
        raise FileNotFoundError(
            "The following processed dataset files are missing:\n"
            f"{formatted_missing}"
        )

    print("=" * 72)
    print("GLOBAL NORMALIZATION EXPERIMENT")
    print("=" * 72)
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Processed dir: {processed_dir}")
    print(f"Output file:   {output_path}")
    print()

    set_seed(SEED)

    print("Loading raw, unnormalized beats...")

    X = np.load(required_files["beats"])
    y = np.load(required_files["labels"])
    record_ids = np.load(required_files["record_ids"])

    if len(X) != len(y) or len(X) != len(record_ids):
        raise ValueError(
            "X_beats.npy, y_labels.npy, and record_ids.npy have "
            "inconsistent lengths."
        )

    if X.ndim != 2:
        raise ValueError(
            f"Expected X with shape (n_beats, beat_length), received {X.shape}."
        )

    print(f"Loaded beats:       {len(X):,}")
    print(f"Beat length:        {X.shape[1]}")
    print(f"Normal beats:       {int(np.sum(y == 0)):,}")
    print(f"Anomalous beats:    {int(np.sum(y == 1)):,}")
    print(f"Unique records:     {len(np.unique(record_ids))}")
    print()

    print("Creating raw random split before normalization...")

    (
        X_train_random,
        X_test_random,
        y_test_random,
        train_records_random,
        test_records_random,
    ) = create_random_split(
        X=X,
        y=y,
        record_ids=record_ids,
    )

    print("Creating raw inter-patient split before normalization...")

    (
        X_train_interpatient,
        X_test_interpatient,
        y_test_interpatient,
        train_records_interpatient,
        test_records_interpatient,
    ) = create_interpatient_split(
        X=X,
        y=y,
        record_ids=record_ids,
    )

    device = engine.get_device()

    print(f"Using device: {device}")

    random_result = train_and_evaluate(
        split_name="random",
        X_train_raw=X_train_random,
        X_test_raw=X_test_random,
        y_test=y_test_random,
        train_record_ids=train_records_random,
        test_record_ids=test_records_random,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    interpatient_result = train_and_evaluate(
        split_name="interpatient",
        X_train_raw=X_train_interpatient,
        X_test_raw=X_test_interpatient,
        y_test=y_test_interpatient,
        train_record_ids=train_records_interpatient,
        test_record_ids=test_records_interpatient,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    random_auc = random_result["metrics"]["auc"]
    interpatient_auc = interpatient_result["metrics"]["auc"]
    auc_gap = random_auc - interpatient_auc

    both_directions_valid = bool(
        random_auc > 0.5
        and interpatient_auc > 0.5
    )

    expected_gap_direction = bool(auc_gap > 0.0)

    final_output = {
        "experiment": "global_normalization_experiment",
        "description": (
            "Comparison of naive random and patient-independent splits using "
            "global z-score normalization fitted only on each split's normal "
            "training beats."
        ),
        "primary_conditions": {
            "condition_a_detector_direction_valid": {
                "criterion": (
                    "AUC must be greater than 0.5 for both random and "
                    "inter-patient splits."
                ),
                "satisfied": both_directions_valid,
            },
            "condition_b_random_auc_above_interpatient_auc": {
                "criterion": (
                    "Random-split AUC must be greater than inter-patient AUC."
                ),
                "satisfied": expected_gap_direction,
            },
        },
        "results": {
            "random": random_result,
            "interpatient": interpatient_result,
        },
        "comparison": {
            "random_auc": float(random_auc),
            "interpatient_auc": float(interpatient_auc),
            "random_minus_interpatient_auc": float(auc_gap),
            "detector_valid_in_both_splits": both_directions_valid,
            "gap_has_expected_direction": expected_gap_direction,
        },
        "decision": {
            "global_normalization_candidate_successful": bool(
                both_directions_valid
                and expected_gap_direction
            ),
            "interpretation": (
                "The candidate is considered successful only if both AUC "
                "values are above 0.5 and the random-split AUC is greater "
                "than the inter-patient AUC."
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            final_output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)
    print(f"Random AUC:                    {random_auc:.4f}")
    print(f"Inter-patient AUC:             {interpatient_auc:.4f}")
    print(f"Random minus inter-patient:    {auc_gap:+.4f}")
    print(
        "AUC > 0.5 for both splits:    "
        f"{both_directions_valid}"
    )
    print(
        "Gap in expected direction:    "
        f"{expected_gap_direction}"
    )
    print(
        "Global candidate successful:  "
        f"{final_output['decision']['global_normalization_candidate_successful']}"
    )
    print()
    print(f"Saved complete output to:\n{output_path}")


if __name__ == "__main__":
    main()