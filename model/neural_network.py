"""
model/neural_network.py
=======================
Defines the Multi-Layer Perceptron (MLP) used for disease risk prediction.
The same architecture is used on every hospital client and on the central server,
so weights can be averaged without shape mismatches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseasePredictionMLP(nn.Module):
    """
    A 4-layer MLP for binary classification (disease risk: 0 = low, 1 = high).

    Architecture
    ------------
    Input  → FC(64) → BN → ReLU → Dropout
           → FC(128) → BN → ReLU → Dropout
           → FC(64)  → BN → ReLU → Dropout
           → FC(32)  → ReLU
           → FC(1)   → Sigmoid
    """

    def __init__(self, input_dim: int = 21, dropout_rate: float = 0.3):
        super(DiseasePredictionMLP, self).__init__()

        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        # --- Layer definitions ---
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(p=dropout_rate)

        # Weight initialisation (Xavier uniform → stable training)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        # Block 2
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        # Block 3
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        # Block 4
        x = F.relu(self.fc4(x))
        # Output (sigmoid for binary probability)
        x = torch.sigmoid(self.fc5(x))
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of positive class (inference mode)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_weights(self) -> dict:
        """Return a copy of all parameter tensors (CPU, detached)."""
        return {k: v.cpu().detach().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict):
        """Load weights from a state-dict-like dictionary."""
        self.load_state_dict(weights)


def count_parameters(model: nn.Module) -> int:
    """Utility: count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = DiseasePredictionMLP(input_dim=21)
    dummy = torch.randn(8, 21)
    out = model(dummy)
    print(f"Output shape : {out.shape}")          # (8, 1)
    print(f"Trainable params: {count_parameters(model):,}")
