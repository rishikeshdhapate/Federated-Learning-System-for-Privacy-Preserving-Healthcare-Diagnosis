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
from sklearn.model_selection import train_test_split   # ✅ ADDED

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.neural_network import DiseasePredictionMLP
from model.metrics import evaluate_model


# ── Differential Privacy helpers ─────────────────────────────────────────────

def _clip_gradients(model: nn.Module, max_norm: float):
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
    std = noise_multiplier * max_norm / batch_size
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.randn_like(p.grad.data) * std
            p.grad.data.add_(noise)


# ── HospitalClient ─────────────────────────────────────────────────────────

class HospitalClient:

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

        self.model = DiseasePredictionMLP(input_dim=input_dim).to(self.device)

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_data(self):
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        def to_tensor(arr):
            return torch.tensor(arr, dtype=torch.float32)

        # ✅ FIX: Create validation split manually
        X = data["X_train"]
        y = data["y_train"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = to_tensor(X_train)
        y_train = to_tensor(y_train)
        X_val   = to_tensor(X_val)
        y_val   = to_tensor(y_val)

        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size, shuffle=True
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
        self.model.set_weights(copy.deepcopy(global_weights))

    def train_local(self) -> dict:
        self.model.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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

        metrics = evaluate_model(self.model, self.val_loader, device=str(self.device))
        metrics["train_loss"] = round(float(np.mean(epoch_losses)), 6)
        metrics["hospital_id"] = self.hospital_id
        metrics["n_train"] = self.n_train
        metrics["dp_enabled"] = self.dp_enabled

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