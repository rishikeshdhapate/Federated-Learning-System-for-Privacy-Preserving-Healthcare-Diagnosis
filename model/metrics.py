"""
model/metrics.py
================
Evaluation helpers used by both clients and the server to measure
model quality after every federated round.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import torch


def evaluate_model(model, data_loader, device="cpu"):
    """
    Run inference over a DataLoader and return a metrics dict.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc, loss,
                    confusion_matrix (2×2 list)
    """
    model.eval()
    criterion = torch.nn.BCELoss()

    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            probs = model(X_batch).squeeze()
            loss = criterion(probs, y_batch.float())

            total_loss += loss.item()
            n_batches += 1

            preds = (probs >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc":       round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.5,
        "loss":      round(total_loss / max(n_batches, 1), 6),
        "confusion_matrix": cm,
    }
    return metrics


def aggregate_metrics(metrics_list: list) -> dict:
    """
    Average a list of metric dicts (one per client).
    Confusion matrices are summed element-wise.
    """
    if not metrics_list:
        return {}

    keys = [k for k in metrics_list[0] if k != "confusion_matrix"]
    aggregated = {}

    for k in keys:
        aggregated[k] = round(float(np.mean([m[k] for m in metrics_list])), 4)

    # Sum confusion matrices
    cm_sum = np.zeros((2, 2), dtype=int)
    for m in metrics_list:
        if "confusion_matrix" in m:
            cm_sum += np.array(m["confusion_matrix"])
    aggregated["confusion_matrix"] = cm_sum.tolist()

    return aggregated
