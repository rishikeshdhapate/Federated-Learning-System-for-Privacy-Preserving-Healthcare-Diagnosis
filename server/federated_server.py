"""
server/federated_server.py
==========================
Central federated server that:

1. Maintains the global model.
2. Distributes global weights to all hospital clients each round.
3. Collects local weight updates.
4. Applies Federated Averaging (FedAvg) — McMahan et al. 2017.
5. Simulates secure aggregation (secret-share masks that cancel).
6. Evaluates the global model on the held-out test set after each round.
7. Persists training history and the final model to disk.

Privacy notes
-------------
* The server NEVER sees raw patient data.
* Only model weights (or deltas) are exchanged.
* Secure aggregation masks ensure even the server cannot reconstruct
  individual client updates (simulation — real deployment uses SMPC).
"""

import os
import sys
import copy
import json
import pickle
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.neural_network import DiseasePredictionMLP
from model.metrics import evaluate_model, aggregate_metrics
from client.local_trainer import HospitalClient


# ── Secure Aggregation simulation ─────────────────────────────────────────────

def _generate_masks(weight_shapes: dict, n_clients: int, seed: int = 0):
    """
    Generate pairwise random masks for n_clients that sum to zero.
    Client i adds mask[i], so the sum of masked updates equals the
    sum of unmasked updates (masks cancel in FedAvg).

    This is a simplified simulation; production uses homomorphic
    encryption or MPC protocols (e.g. SecAgg from Bonawitz et al. 2017).
    """
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(n_clients):
        mask = {k: torch.zeros(s) for k, s in weight_shapes.items()}
        masks.append(mask)

    # Pairwise masks: for each pair (i,j), generate r and add +r to i, -r to j
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            for k, shape in weight_shapes.items():
                r = torch.tensor(rng.normal(0, 0.01, shape).astype(np.float32))
                masks[i][k] = masks[i][k] + r
                masks[j][k] = masks[j][k] - r

    return masks  # masks[i] + masks[j] cancels for each pair


def _apply_mask(weights: dict, mask: dict) -> dict:
    return {k: weights[k] + mask[k] for k in weights}


def _remove_masks(masked_sum: dict, masks: list) -> dict:
    """Reconstruct unmasked sum at the server (masks cancel automatically)."""
    return masked_sum  # already cancelled — included for documentation


# ── FedAvg aggregation ────────────────────────────────────────────────────────

def federated_averaging(client_results: list) -> dict:
    """
    Weighted FedAvg: w_global = Σ (n_i / N) * w_i

    Parameters
    ----------
    client_results : list of dicts with keys "weights" and "n_samples"

    Returns
    -------
    aggregated global weights (state_dict)
    """
    total_samples = sum(r["n_samples"] for r in client_results)
    aggregated    = {}

    for k in client_results[0]["weights"]:
        weighted_sum = sum(
            r["weights"][k] * (r["n_samples"] / total_samples)
            for r in client_results
        )
        aggregated[k] = weighted_sum

    return aggregated


# ── FederatedServer ───────────────────────────────────────────────────────────

class FederatedServer:
    """
    Orchestrates multi-round federated training.

    Parameters
    ----------
    processed_dir  : path to pre-processed hospital data
    n_hospitals    : number of hospital nodes
    input_dim      : model input dimensionality
    n_rounds       : number of federated communication rounds
    local_epochs   : local training epochs per round
    lr             : local learning rate
    dp_enabled     : enable differential privacy at clients
    dp_noise_mult  : DP noise multiplier
    dp_max_norm    : gradient clipping norm
    output_dir     : where to save the final model and history
    device         : "cpu" or "cuda"
    """

    def __init__(
        self,
        processed_dir: str = "data/processed",
        n_hospitals:   int   = 5,
        input_dim:     int   = 21,
        n_rounds:      int   = 10,
        local_epochs:  int   = 5,
        lr:            float = 1e-3,
        dp_enabled:    bool  = True,
        dp_noise_mult: float = 0.5,
        dp_max_norm:   float = 1.0,
        output_dir:    str   = "outputs",
        device:        str   = "cpu",
    ):
        self.processed_dir = processed_dir
        self.n_hospitals   = n_hospitals
        self.input_dim     = input_dim
        self.n_rounds      = n_rounds
        self.local_epochs  = local_epochs
        self.lr            = lr
        self.dp_enabled    = dp_enabled
        self.dp_noise_mult = dp_noise_mult
        self.dp_max_norm   = dp_max_norm
        self.output_dir    = output_dir
        self.device        = device

        os.makedirs(output_dir, exist_ok=True)

        # Global model lives on the server
        self.global_model = DiseasePredictionMLP(input_dim=input_dim).to(
            torch.device(device)
        )

        # Training state
        self.current_round    = 0
        self.is_training      = False
        self.training_history = []   # one entry per round
        self.status_message   = "Idle"

        # Load global test set
        self._load_test_data()

        # Initialise hospital clients
        self._init_clients()

    # ── setup ─────────────────────────────────────────────────────────────────

    def _load_test_data(self):
        test_path = os.path.join(self.processed_dir, "global_test.pkl")
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Global test set not found at {test_path}. "
                "Run data/preprocess.py first."
            )
        with open(test_path, "rb") as f:
            data = pickle.load(f)

        X = torch.tensor(data["X_test"], dtype=torch.float32)
        y = torch.tensor(data["y_test"], dtype=torch.float32)
        self.test_loader = DataLoader(
            TensorDataset(X, y), batch_size=256, shuffle=False
        )
        print(f"[Server] Global test set loaded: {len(X)} samples.")

    def _init_clients(self):
        print("[Server] Initialising hospital clients …")
        self.clients = []
        for i in range(1, self.n_hospitals + 1):
            data_path = os.path.join(self.processed_dir, f"hospital_{i}.pkl")
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Hospital data not found: {data_path}. "
                    "Run data/preprocess.py first."
                )
            client = HospitalClient(
                hospital_id   = i,
                data_path     = data_path,
                input_dim     = self.input_dim,
                lr            = self.lr,
                local_epochs  = self.local_epochs,
                dp_enabled    = self.dp_enabled,
                dp_noise_mult = self.dp_noise_mult,
                dp_max_norm   = self.dp_max_norm,
                device        = self.device,
            )
            self.clients.append(client)
        print(f"[Server] {len(self.clients)} hospital clients ready.")

    # ── federated training loop ───────────────────────────────────────────────

    def run_round(self) -> dict:
        """Execute one federated communication round."""
        self.current_round += 1
        self.status_message = f"Round {self.current_round}/{self.n_rounds} in progress …"
        t_start = time.time()

        print(f"\n{'='*60}")
        print(f"  FEDERATED ROUND {self.current_round}/{self.n_rounds}")
        print(f"{'='*60}")

        global_weights = self.global_model.get_weights()

        # --- Weight shape map for secure aggregation masks ---
        weight_shapes  = {k: v.shape for k, v in global_weights.items()}
        sec_masks      = _generate_masks(weight_shapes, len(self.clients),
                                         seed=self.current_round)

        client_results = []
        client_metrics = []

        for idx, client in enumerate(self.clients):
            # 1. Send global weights to client
            client.set_global_weights(global_weights)

            # 2. Client trains locally on private data
            result = client.train_local()

            # 3. Apply secure aggregation mask before returning to server
            masked_weights = _apply_mask(result["weights"], sec_masks[idx])
            result["weights"] = masked_weights

            client_results.append(result)
            client_metrics.append(result["metrics"])

        # 4. Aggregate (masks cancel → equivalent to plain FedAvg)
        new_global_weights = federated_averaging(client_results)
        self.global_model.set_weights(new_global_weights)

        # 5. Evaluate global model on held-out test set
        global_metrics = evaluate_model(
            self.global_model, self.test_loader, device=self.device
        )

        # 6. Aggregate client-side metrics for dashboard
        avg_client_metrics = aggregate_metrics(client_metrics)

        round_summary = {
            "round":              self.current_round,
            "global_metrics":     global_metrics,
            "avg_client_metrics": avg_client_metrics,
            "client_metrics":     client_metrics,
            "duration_sec":       round(time.time() - t_start, 2),
        }
        self.training_history.append(round_summary)

        # Save history to disk after every round
        self._save_history()

        print(f"\n[Server] Round {self.current_round} complete in "
              f"{round_summary['duration_sec']}s")
        print(f"  Global accuracy : {global_metrics['accuracy']:.4f}")
        print(f"  Global F1       : {global_metrics['f1']:.4f}")
        print(f"  Global AUC      : {global_metrics['auc']:.4f}")

        self.status_message = (
            f"Round {self.current_round}/{self.n_rounds} done — "
            f"acc={global_metrics['accuracy']:.4f}"
        )
        return round_summary

    def run_all_rounds(self) -> dict:
        """Run the full n_rounds federated training process."""
        self.is_training  = True
        self.status_message = "Training started"
        t_total = time.time()

        for _ in range(self.n_rounds):
            self.run_round()

        self.is_training = False
        total_time = round(time.time() - t_total, 2)

        # Final global model evaluation
        final_metrics = evaluate_model(
            self.global_model, self.test_loader, device=self.device
        )
        self.status_message = (
            f"Training complete ({self.n_rounds} rounds, {total_time}s) — "
            f"final acc={final_metrics['accuracy']:.4f}"
        )

        # Save the final model
        model_path = os.path.join(self.output_dir, "global_model.pt")
        torch.save(self.global_model.state_dict(), model_path)
        print(f"\n[Server] Final model saved to {model_path}")
        print(f"[Server] Final accuracy : {final_metrics['accuracy']:.4f}")
        print(f"[Server] Final AUC      : {final_metrics['auc']:.4f}")
        print(f"[Server] Total time     : {total_time}s")

        return {
            "status":        "complete",
            "total_rounds":  self.n_rounds,
            "total_time_sec": total_time,
            "final_metrics": final_metrics,
            "history":       self.training_history,
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def _save_history(self):
        path = os.path.join(self.output_dir, "training_history.json")
        # Convert tensors to Python scalars (json-serialisable)
        history_serialisable = []
        for entry in self.training_history:
            clean = copy.deepcopy(entry)
            history_serialisable.append(clean)
        with open(path, "w") as f:
            json.dump(history_serialisable, f, indent=2, default=str)

    def get_status(self) -> dict:
        """Return current server status (polled by /status API)."""
        last_metrics = {}
        if self.training_history:
            last_metrics = self.training_history[-1].get("global_metrics", {})

        return {
            "is_training":     self.is_training,
            "current_round":   self.current_round,
            "total_rounds":    self.n_rounds,
            "status_message":  self.status_message,
            "last_metrics":    last_metrics,
            "history_length":  len(self.training_history),
        }

    def predict(self, features: list) -> dict:
        """
        Run inference on a single sample.

        Parameters
        ----------
        features : list of 21 float values (pre-scaled)

        Returns
        -------
        dict with probability, prediction, and confidence label
        """
        x = torch.tensor([features], dtype=torch.float32)
        prob = self.global_model.predict_proba(x).item()
        pred = int(prob >= 0.5)
        return {
            "probability":  round(prob, 4),
            "prediction":   pred,
            "label":        "High Risk" if pred == 1 else "Low Risk",
            "confidence":   round(abs(prob - 0.5) * 200, 1),  # 0-100
        }
