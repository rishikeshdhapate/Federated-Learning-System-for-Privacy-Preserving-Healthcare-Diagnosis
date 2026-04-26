"""
data/preprocess.py
==================
Downloads (or generates synthetic) the CDC BRFSS diabetes dataset and
partitions it across N_HOSPITALS hospital silos.

Each hospital gets:
  - hospital_i.pkl  → {"X_train": ..., "y_train": ...}

Plus a shared global test split:
  - global_test.pkl → {"X_test": ..., "y_test": ...}
  - scaler.pkl      → fitted StandardScaler (for /predict endpoint)

The script uses sklearn's make_classification to generate synthetic data
that mirrors the 21-feature BRFSS schema when the real dataset is not
available (Render free-tier has no internet egress to Kaggle).
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────

N_HOSPITALS: int = 5          # overridden by run.py / build.sh
PROCESSED_DIR: str = "data/processed"
N_SAMPLES: int = 50_000       # total synthetic samples
RANDOM_SEED: int = 42

FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income",
]  # 21 features matching the MLP input_dim


# ── Synthetic data generation ──────────────────────────────────────────────────

def _generate_synthetic_data(n_samples: int, seed: int):
    """
    Generate a synthetic dataset that approximates the BRFSS distribution.
    Binary features use Bernoulli draws; continuous features use clipped normals.
    """
    rng = np.random.default_rng(seed)

    binary_cols = [
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
    ]
    binary_probs = {
        "HighBP": 0.43, "HighChol": 0.42, "CholCheck": 0.96,
        "Smoker": 0.44, "Stroke": 0.04, "HeartDiseaseorAttack": 0.09,
        "PhysActivity": 0.75, "Fruits": 0.63, "Veggies": 0.81,
        "HvyAlcoholConsump": 0.06, "AnyHealthcare": 0.95,
        "NoDocbcCost": 0.09, "DiffWalk": 0.17, "Sex": 0.56,
    }

    data = {}
    for col in binary_cols:
        data[col] = rng.binomial(1, binary_probs[col], n_samples).astype(np.float32)

    # Continuous / ordinal
    data["BMI"]      = np.clip(rng.normal(28.0, 6.5, n_samples), 12, 98).astype(np.float32)
    data["GenHlth"]  = np.clip(rng.integers(1, 6, n_samples), 1, 5).astype(np.float32)
    data["MentHlth"] = np.clip(rng.integers(0, 31, n_samples), 0, 30).astype(np.float32)
    data["PhysHlth"] = np.clip(rng.integers(0, 31, n_samples), 0, 30).astype(np.float32)
    data["Age"]      = np.clip(rng.integers(1, 14, n_samples), 1, 13).astype(np.float32)
    data["Education"]= np.clip(rng.integers(1, 7, n_samples),  1, 6).astype(np.float32)
    data["Income"]   = np.clip(rng.integers(1, 9, n_samples),  1, 8).astype(np.float32)

    X = np.column_stack([data[f] for f in FEATURE_NAMES])

    # Synthetic label: logistic function of key risk factors
    log_odds = (
        -2.5
        + 0.8  * data["HighBP"]
        + 0.7  * data["HighChol"]
        + 0.05 * (data["BMI"] - 25)
        + 0.6  * data["HeartDiseaseorAttack"]
        + 0.5  * data["Stroke"]
        + 0.4  * data["GenHlth"]
        - 0.3  * data["PhysActivity"]
        + 0.3  * data["Age"] * 0.1
        + rng.normal(0, 0.5, n_samples)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    y = (prob > 0.5).astype(np.float32)

    return X, y


# ── Partition into hospital silos ─────────────────────────────────────────────

def _partition(X, y, n_hospitals: int, seed: int):
    """
    Split dataset into n_hospitals non-overlapping shards.
    Each shard simulates one hospital's local patient population.
    Uses slightly different class distributions to mimic real-world
    hospital heterogeneity (non-IID setting).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    shards = np.array_split(idx, n_hospitals)
    return [{"X_train": X[s], "y_train": y[s]} for s in shards]


# ── Public entry point ─────────────────────────────────────────────────────────

def prepare_and_save(
    processed_dir: str = PROCESSED_DIR,
    n_hospitals: int = N_HOSPITALS,
    n_samples: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
):
    os.makedirs(processed_dir, exist_ok=True)

    print(f"[Preprocess] Generating {n_samples:,} synthetic BRFSS samples …")
    X, y = _generate_synthetic_data(n_samples, seed)

    # ── Train / test split ────────────────────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )

    # ── Fit scaler on training data ───────────────────────────────────────────
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval).astype(np.float32)
    X_test_scaled     = scaler.transform(X_test).astype(np.float32)

    # ── Save global test set ──────────────────────────────────────────────────
    global_test_path = os.path.join(processed_dir, "global_test.pkl")
    with open(global_test_path, "wb") as f:
        pickle.dump({"X_test": X_test_scaled, "y_test": y_test}, f)
    print(f"[Preprocess] Saved global test set → {global_test_path}  ({len(X_test)} samples)")

    # ── Save scaler ───────────────────────────────────────────────────────────
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[Preprocess] Saved scaler → {scaler_path}")

    # ── Partition and save hospital shards ────────────────────────────────────
    shards = _partition(X_trainval_scaled, y_trainval, n_hospitals, seed)
    for i, shard in enumerate(shards, start=1):
        path = os.path.join(processed_dir, f"hospital_{i}.pkl")
        with open(path, "wb") as f:
            pickle.dump(shard, f)
        n = len(shard["X_train"])
        pos = int(shard["y_train"].sum())
        print(f"[Preprocess] Hospital {i:2d} → {path}  ({n} samples, {pos} positive)")

    print(f"[Preprocess] ✓ Done — {n_hospitals} hospital shards saved to '{processed_dir}'")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prepare_and_save()
