"""
Ablation experiment: isolates whether per-patient normalization (introduced
to fix low recall) is responsible for neutralizing the random-vs-interpatient
data leakage gap that is the project's core methodological contribution.

Re-generates train/test splits using the ORIGINAL per-beat normalization
(each beat normalized by its own mean/std, not the patient's), trains the
same Conv1D architecture on both random and inter-patient splits, and
compares the resulting AUC gap against the one already measured with
per-patient normalization (stored in results/all_results.json).

Does not modify the main pipeline, its checkpoints, or its results file.
"""

import os
import json
import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

import src.engine as engine
from src.models.conv_autoencoder import ConvAutoencoder

# Same DS1/DS2 split used everywhere else in the project
DS1_RECORDS = {'101','106','108','109','112','114','115','116','118','119',
               '122','124','201','203','205','207','208','209','215','220',
               '223','230'}
DS2_RECORDS = {'100','103','105','111','113','117','121','123','200','202',
               '210','212','213','214','219','221','222','228','231','232',
               '233','234'}

EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


def normalize_per_beat(X):
    """ORIGINAL normalization: each beat normalized by its own mean/std."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1e-8
    return (X - mean) / std


def random_split(X, y, test_size=0.2, random_state=42):
    X_normal, X_anomalous = X[y == 0], X[y == 1]
    X_train, X_test_normal = train_test_split(X_normal, test_size=test_size, random_state=random_state)
    X_test = np.concatenate([X_test_normal, X_anomalous], axis=0)
    y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_anomalous))])
    return X_train, X_test, y_test


def inter_patient_split(X, y, record_ids):
    ds1_mask = np.isin(record_ids, list(DS1_RECORDS))
    ds2_mask = np.isin(record_ids, list(DS2_RECORDS))
    X_train = X[ds1_mask & (y == 0)]
    X_test = X[ds2_mask]
    y_test = y[ds2_mask]
    return X_train, X_test, y_test


def train_and_eval(X_train, X_test, y_test, device, run_label):
    X_train_t = torch.from_numpy(X_train.reshape(-1, 1, X_train.shape[1]).astype(np.float32))
    X_test_t = torch.from_numpy(X_test.reshape(-1, 1, X_test.shape[1]).astype(np.float32))

    model = ConvAutoencoder(bottleneck_channels=64)
    print(f"  Training {run_label}...")
    model, _ = engine.train(model, X_train_t, device, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    errors = engine.compute_errors(model, X_test_t, device)
    metrics = engine.evaluate(errors, y_test)
    print(f"  {run_label}: AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f}")
    return metrics


def main():
    set_seed(42)

    with open("configs/experiments.yaml") as f:
        config = yaml.safe_load(f)

    processed_dir = config["project"]["processed_dir"]
    device = engine.get_device()
    print(f"Using device: {device}\n")

    print("Loading raw (un-normalized) beats...")
    X = np.load(os.path.join(processed_dir, "X_beats.npy"))
    y = np.load(os.path.join(processed_dir, "y_labels.npy"))
    record_ids = np.load(os.path.join(processed_dir, "record_ids.npy"))

    print("Applying PER-BEAT normalization (original, pre-fix method)...")
    X_norm = normalize_per_beat(X)

    print("\n=== Random split (per-beat normalization) ===")
    X_train_r, X_test_r, y_test_r = random_split(X_norm, y)
    metrics_random = train_and_eval(X_train_r, X_test_r, y_test_r, device, "conv_random_perbeat")

    print("\n=== Inter-patient split (per-beat normalization) ===")
    X_train_ip, X_test_ip, y_test_ip = inter_patient_split(X_norm, y, record_ids)
    metrics_ip = train_and_eval(X_train_ip, X_test_ip, y_test_ip, device, "conv_interpatient_perbeat")

    gap_perbeat = metrics_random["auc"] - metrics_ip["auc"]

    # Load current per-patient-normalization results for direct comparison
    results_path = os.path.join(config["project"]["project_dir"], "results", "all_results.json")
    with open(results_path) as f:
        current_results = json.load(f)
    gap_perpatient = current_results["conv_random"]["auc"] - current_results["conv_interpatient"]["auc"]

    print("\n" + "=" * 60)
    print("ABLATION RESULT: random-split AUC minus inter-patient AUC")
    print("=" * 60)
    print(f"With PER-BEAT normalization   : {gap_perbeat:+.4f}  "
          f"(random={metrics_random['auc']:.4f}, interpatient={metrics_ip['auc']:.4f})")
    print(f"With PER-PATIENT normalization: {gap_perpatient:+.4f}  "
          f"(random={current_results['conv_random']['auc']:.4f}, "
          f"interpatient={current_results['conv_interpatient']['auc']:.4f})")

    out = {
        "per_beat_normalization": {"random": metrics_random, "interpatient": metrics_ip, "gap": gap_perbeat},
        "per_patient_normalization_reference": {
            "random_auc": current_results["conv_random"]["auc"],
            "interpatient_auc": current_results["conv_interpatient"]["auc"],
            "gap": gap_perpatient,
        },
    }
    out_path = os.path.join(config["project"]["project_dir"], "results", "normalization_ablation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()