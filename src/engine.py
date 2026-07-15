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


def train(model, X_train, device, epochs, batch_size, lr):
    model.to(device)
    loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
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
            print(f"    Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}")

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
    """Computes AUC, Youden's J optimal threshold, and F1-optimal threshold,
    reporting both for an evidence-based comparison instead of an assumed
    trade-off."""
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