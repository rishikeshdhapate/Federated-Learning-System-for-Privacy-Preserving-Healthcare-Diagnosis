"""
client/local_trainer.py
=======================
Simulates a single hospital client in the federated learning system.

Responsibilities
----------------
1. Load the local hospital dataset (never shared with the server).
2. Initialise / receive the global model weights.
3. Train the local model for a fixed number of epochs.
4. Apply Differential Privacy (Gaussian noise on gradients) before
   returning weight updates to the server.
5. Return only the weight deltas (Δw), NOT the raw data.

Privacy Guarantees
------------------
* Raw data never leaves this module.
* Gradient noise (σ = dp_noise_multiplier × C) is added after each
  backward pass (per-sample gradient clipping + noise addition).
* This is a simulation of DP-SGD; for production use the
  Opacus library (opacus.ai) with formal (ε, δ)-DP accounting.
"""

import os
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Adjust import path when running as a module inside the package
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.neural_network import DiseasePredictionMLP
from model.metrics import evaluate_model


# ── Differential Privacy helpers ─────────────────────────────────────────────

def _clip_gradients(model: nn.Module, max_norm: float):
    """Clip per-parameter gradients to L2 norm ≤ max_norm (sensitivity bound C)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    return total_norm


def _add_gaussian_noise(model: nn.Module, noise_multiplier: float,
                        max_norm: float, batch_size: int):
    """
    Add calibrated Gaussian noise to gradients.
    Noise std = noise_multiplier × max_norm / batch_size
    """
    std = noise_multiplier * max_norm / batch_size
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.randn_like(p.grad.data) * std
            p.grad.data.add_(noise)


# ── HospitalClient ─────────────────────────────────────────────────────────

class HospitalClient:
    """
    Represents a single hospital participating in federated learning.

    Parameters
    ----------
    hospital_id   : int   – unique node identifier
    data_path     : str   – path to the pre-processed hospital .pkl file
    input_dim     : int   – number of input features
    lr            : float – local learning rate
    local_epochs  : int   – epochs per federated round
    batch_size    : int   – mini-batch size
    dp_enabled    : bool  – whether to apply differential privacy
    dp_noise_mult : float – DP noise multiplier (σ / C)
    dp_max_norm   : float – gradient clipping norm (sensitivity C)
    device        : str   – "cpu" or "cuda"
    """

    def __init__(
        self,
        hospital_id: int,
        data_path: str,
        input_dim: int = 21,
        lr: float = 1e-3,
        local_epochs: int = 5,
        batch_size: int = 32,
        dp_enabled: bool = True,
        dp_noise_mult: float = 0.5,
        dp_max_norm: float = 1.0,
        device: str = "cpu",
    ):
        self.hospital_id   = hospital_id
        self.data_path     = data_path
        self.input_dim     = input_dim
        self.lr            = lr
        self.local_epochs  = local_epochs
        self.batch_size    = batch_size
        self.dp_enabled    = dp_enabled
        self.dp_noise_mult = dp_noise_mult
        self.dp_max_norm   = dp_max_norm
        self.device        = torch.device(device)

        self._load_data()

        # Build local model (will receive global weights each round)
        self.model = DiseasePredictionMLP(input_dim=input_dim).to(self.device)

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_data(self):
        """Load pre-split hospital dataset from disk.  Data never leaves this fn."""
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        def to_tensor(arr):
            return torch.tensor(arr, dtype=torch.float32)

        X_train = to_tensor(data["X_train"])
        y_train = to_tensor(data["y_train"])
        X_val   = to_tensor(data["X_val"])
        y_val   = to_tensor(data["y_val"])

        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        self.val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=256, shuffle=False
        )
        self.n_train = len(X_train)
        self.n_val   = len(X_val)

        print(f"  [Hospital {self.hospital_id}] Loaded "
              f"{self.n_train} train / {self.n_val} val samples.")

    # ── federated round ───────────────────────────────────────────────────────

    def set_global_weights(self, global_weights: dict):
        """Receive and load the latest global model weights from the server."""
        self.model.set_weights(copy.deepcopy(global_weights))

    def train_local(self) -> dict:
        """
        Run local_epochs of training on the hospital's private data.

        Returns
        -------
        result dict:
            weights  : updated model state_dict
            delta    : weight deltas (w_local - w_global)
            metrics  : val-set evaluation metrics
            n_samples: number of training samples (for weighted aggregation)
        """
        self.model.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Keep a snapshot of the weights BEFORE local training (for delta)
        global_weights_snapshot = copy.deepcopy(self.model.state_dict())

        epoch_losses = []

        for epoch in range(self.local_epochs):
            running_loss = 0.0
            n_batches    = 0

            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.model(X_batch).squeeze()
                loss  = criterion(preds, y_batch)
                loss.backward()

                # ── Differential Privacy ─────────────────────────────────────
                if self.dp_enabled:
                    _clip_gradients(self.model, self.dp_max_norm)
                    _add_gaussian_noise(
                        self.model, self.dp_noise_mult,
                        self.dp_max_norm, X_batch.size(0)
                    )

                optimizer.step()
                running_loss += loss.item()
                n_batches    += 1

            avg_loss = running_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

        # Evaluate on local validation set
        metrics = evaluate_model(self.model, self.val_loader, device=str(self.device))
        metrics["train_loss"] = round(float(np.mean(epoch_losses)), 6)
        metrics["hospital_id"] = self.hospital_id
        metrics["n_train"] = self.n_train
        metrics["dp_enabled"] = self.dp_enabled

        # Compute weight deltas (used by secure aggregation simulation)
        local_weights = self.model.get_weights()
        delta_weights = {
            k: local_weights[k] - global_weights_snapshot[k].cpu()
            for k in local_weights
        }

        print(
            f"  [Hospital {self.hospital_id}] "
            f"acc={metrics['accuracy']:.4f}  "
            f"loss={metrics['train_loss']:.4f}  "
            f"f1={metrics['f1']:.4f}  "
            f"dp={'ON' if self.dp_enabled else 'OFF'}"
        )

        return {
            "hospital_id": self.hospital_id,
            "weights":     local_weights,
            "delta":       delta_weights,
            "metrics":     metrics,
            "n_samples":   self.n_train,
        }
