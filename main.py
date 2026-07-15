"""
Single entry point for the entire ECG autoencoder anomaly detection study.

Runs, in one pass:
  1. Data preparation (download + beat extraction)
  2. Preprocessing (normalization + random/inter-patient splits)
  3. Every unique model configuration listed in configs/experiments.yaml,
     training each exactly once and skipping if a checkpoint already exists
  4. Evaluation of every run
  5. Consolidated comparison tables/plots (architecture, bottleneck study)

Usage:
    python main.py
"""

import os
import json
import random
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

import src.data_prep as data_prep
import src.preprocessing as preprocessing
import src.engine as engine


def set_seed(seed=42):
    """
    Fixes all sources of randomness (Python, NumPy, PyTorch CPU/GPU) so that
    re-running the pipeline produces identical results. Without this, weight
    initialization and DataLoader shuffling are non-deterministic, and
    reported metrics can shift noticeably between runs even with identical
    code and data (observed empirically: AUC varied by ~0.06 across runs
    before this fix was introduced).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

def load_config():
    with open("configs/experiments.yaml") as f:
        return yaml.safe_load(f)


def run_single(run_cfg, config, device):
    """Trains (or loads if checkpoint exists) and evaluates a single run."""
    run_id = run_cfg["id"]
    models_dir = os.path.join(config["project"]["project_dir"], "models")
    results_dir = os.path.join(config["project"]["project_dir"], "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = os.path.join(models_dir, f"{run_id}.pt")
    processed_dir = config["project"]["processed_dir"]

    X_train, X_test, y_test = engine.load_split(processed_dir, run_cfg["split"])
    model = engine.build_model(run_cfg)

    if os.path.exists(checkpoint_path):
        print(f"[{run_id}] Checkpoint found, loading (skipping training).")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
    else:
        print(f"[{run_id}] Training from scratch...")
        t_cfg = config["training"]
        model, loss_history = engine.train(
            model, X_train, device,
            epochs=t_cfg["epochs"], batch_size=t_cfg["batch_size"], lr=t_cfg["learning_rate"]
        )
        torch.save(model.state_dict(), checkpoint_path)

        plt.figure(figsize=(7, 4))
        plt.plot(loss_history)
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
        plt.title(f"Training Loss ({run_id})")
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"loss_{run_id}.png"))
        plt.close()

    print(f"[{run_id}] Evaluating...")
    errors = engine.compute_errors(model, X_test, device)
    metrics = engine.evaluate(errors, y_test)
    print(f"[{run_id}] AUC={metrics['auc']:.4f}  Precision={metrics['precision']:.4f}  "
          f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

    return metrics


def make_comparison_outputs(config, all_metrics):
    results_dir = os.path.join(config["project"]["project_dir"], "results")
    comparisons = config.get("comparisons", {})

    # Architecture comparison: simple table (Dense vs Conv1D)
    if "architecture_comparison" in comparisons:
        ids = comparisons["architecture_comparison"]["run_ids"]
        print("\n=== Architecture Comparison ===")
        for rid in ids:
            m = all_metrics[rid]
            print(f"{rid}: AUC={m['auc']:.4f} Precision={m['precision']:.4f} "
                  f"Recall={m['recall']:.4f} F1={m['f1']:.4f}")

    # Bottleneck study: line plot of AUC vs bottleneck width
    if "bottleneck_study" in comparisons:
        ids = comparisons["bottleneck_study"]["run_ids"]
        x_values = comparisons["bottleneck_study"]["x_axis_values"]
        aucs = [all_metrics[rid]["auc"] for rid in ids]

        print("\n=== Bottleneck Sensitivity ===")
        for x, a in zip(x_values, aucs):
            print(f"Bottleneck={x}: AUC={a:.4f}")

        plt.figure(figsize=(7, 5))
        plt.plot(x_values, aucs, marker="o")
        plt.xlabel("Bottleneck channels")
        plt.ylabel("AUC")
        plt.title("Bottleneck Size vs Detection Performance")
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, "bottleneck_sensitivity.png"))
        plt.close()

    # Save every run's metrics, consolidated in a single file
    summary_path = os.path.join(results_dir, "all_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll results saved to {summary_path}")


def main():
    set_seed(42)
    config = load_config()
    device = engine.get_device()
    print(f"Using device: {device}\n")

    print("=== Step 1: Data preparation ===")
    data_prep.run(config)

    print("\n=== Step 2: Preprocessing ===")
    preprocessing.run(config)

    print("\n=== Step 3: Training / evaluating all runs ===")
    all_metrics = {}
    for run_cfg in config["runs"]:
        all_metrics[run_cfg["id"]] = run_single(run_cfg, config, device)

    print("\n=== Step 4: Consolidated comparisons ===")
    make_comparison_outputs(config, all_metrics)

    print("\nDone. Full study complete in a single run.")


if __name__ == "__main__":
    main()