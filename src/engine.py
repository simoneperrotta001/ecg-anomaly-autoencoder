"""
Shared training and evaluation engine, used identically for every run
(regardless of architecture or split) so there is exactly one code path
for "how a model is trained" and "how a model is evaluated" — no
duplicated logic across experiments.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix

from src.models.conv_autoencoder import ConvAutoencoder
from src.models.dense_autoencoder import DenseAutoencoder


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_model(run_cfg):
    if run_cfg["architecture"] == "conv1d":
        return ConvAutoencoder(bottleneck_channels=run_cfg["bottleneck_channels"])
    if run_cfg["architecture"] == "dense":
        return DenseAutoencoder(latent_dim=run_cfg["latent_dim"])
    raise ValueError(f"Unknown architecture: {run_cfg['architecture']}")


def load_split(processed_dir, split_name):
    X_train = np.load(f"{processed_dir}/X_train_{split_name}.npy").astype(np.float32)
    X_test = np.load(f"{processed_dir}/X_test_{split_name}.npy").astype(np.float32)
    y_test = np.load(f"{processed_dir}/y_test_{split_name}.npy")
    X_train = torch.from_numpy(X_train.reshape(-1, 1, X_train.shape[1]))
    X_test = torch.from_numpy(X_test.reshape(-1, 1, X_test.shape[1]))
    return X_train, X_test, y_test


def train(
    model,
    X_train,
    device,
    epochs,
    batch_size,
    lr,
    seed=42,
):
    """
    Train one autoencoder using a deterministic DataLoader order.

    The explicit generator makes batch shuffling independent of the
    experiments executed before this run.
    """
    model.to(device)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        TensorDataset(X_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(X_train)
        loss_history.append(epoch_loss)

        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"    Epoch {epoch}/{epochs} "
                f"- Loss: {epoch_loss:.6f}",
                flush=True,
            )

    return model, loss_history


def compute_errors(model, X_test, device, batch_size=256):
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i + batch_size].to(device)
            reconstruction = model(batch)
            mse = torch.mean((reconstruction - batch) ** 2, dim=(1, 2))
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


def evaluate(errors, y_test):
    """
    Compute ROC-AUC and select an exploratory operating threshold using
    Youden's J statistic.

    Precision, recall, F1, and the confusion matrix are evaluated at the
    test-selected Youden threshold. ROC-AUC remains the primary
    threshold-independent comparison metric.
    """
    fpr, tpr, roc_thresholds = roc_curve(y_test, errors)
    roc_auc = auc(fpr, tpr)

    youden_idx = np.argmax(tpr - fpr)
    youden_threshold = roc_thresholds[youden_idx]
    preds = (errors > youden_threshold).astype(int)

    metrics = {
        "auc": float(roc_auc),
        "threshold_youden": float(youden_threshold),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }

    return metrics

def save_diagnostic_plots(run_id, errors, y_test, metrics, results_dir):
    """
    Saves confusion matrix, ROC curve, PR curve, and error distribution
    plots for a single run, named after its run_id (not split name), so
    every run in configs/experiments.yaml gets its own complete set of
    diagnostic plots.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay
    import numpy as np

    fpr = np.array(metrics["fpr"])
    tpr = np.array(metrics["tpr"])
    threshold = metrics["threshold_youden"]
    cm = np.array(metrics["confusion_matrix"])

    # 1. Error distribution
    plt.figure(figsize=(8, 5))
    plt.hist(errors[y_test == 0], bins=100, alpha=0.6, label="Normal")
    plt.hist(errors[y_test == 1], bins=100, alpha=0.6, label="Anomalous")
    plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
    plt.xlabel("Reconstruction error (MSE)")
    plt.ylabel("Count")
    plt.title(f"Reconstruction Error Distribution ({run_id})")
    plt.legend()
    plt.savefig(f"{results_dir}/error_distribution_{run_id}.png")
    plt.close()

    # 2. ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {metrics['auc']:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({run_id})")
    plt.legend()
    plt.savefig(f"{results_dir}/roc_curve_{run_id}.png")
    plt.close()

    # 3. Precision-Recall curve
    precisions, recalls, _ = precision_recall_curve(y_test, errors)
    plt.figure(figsize=(6, 6))
    plt.plot(recalls, precisions, label="Precision-Recall curve")
    plt.scatter(metrics["recall"], metrics["precision"], color="red", zorder=5,
                label=f"Youden's J point (F1={metrics['f1']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({run_id})")
    plt.legend()
    plt.savefig(f"{results_dir}/pr_curve_{run_id}.png")
    plt.close()

    # 4. Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomalous"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix ({run_id})")
    plt.savefig(f"{results_dir}/confusion_matrix_{run_id}.png")
    plt.close()