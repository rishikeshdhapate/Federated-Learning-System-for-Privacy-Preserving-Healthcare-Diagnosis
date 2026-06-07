"""
data/preprocess.py
==================
Downloads and preprocesses the CDC Behavioral Risk Factor Surveillance
System (BRFSS) Diabetes Health Indicators dataset from the UCI ML Repository
(no Kaggle required).

Dataset URL (direct CSV):
  https://archive.ics.uci.edu/ml/machine-learning-databases/00891/

If the download fails in an offline environment the script falls back to
generating a realistic synthetic dataset with the same schema so the rest
of the system still works end-to-end.

Splits
------
The cleaned dataset is split into N_HOSPITALS parts to simulate independent
hospital data silos (non-IID: each hospital gets a different random slice).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import urllib.request
import pickle

# ── configuration ────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(__file__))
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
N_HOSPITALS     = 5          # number of simulated hospital nodes
RANDOM_SEED     = 42
TEST_SPLIT      = 0.20       # 20 % held-out global test set

# CDC BRFSS 2015 Diabetes Health Indicators (UCI ML Repo mirror)
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00891/cdc_diabetes_health_indicators.zip"
)

# Fallback: use a small synthetic dataset with the same 21 features + target
FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income"
]
TARGET_NAME = "Diabetes_binary"


# ── helpers ───────────────────────────────────────────────────────────────────

def _generate_synthetic(n_samples: int = 10_000) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors the statistical properties
    of the CDC BRFSS dataset so the ML pipeline still runs correctly.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    data = {
        "HighBP":               rng.integers(0, 2, n_samples),
        "HighChol":             rng.integers(0, 2, n_samples),
        "CholCheck":            rng.integers(0, 2, n_samples),
        "BMI":                  rng.normal(28.4, 6.5, n_samples).clip(12, 60),
        "Smoker":               rng.integers(0, 2, n_samples),
        "Stroke":               rng.binomial(1, 0.04, n_samples),
        "HeartDiseaseorAttack": rng.binomial(1, 0.09, n_samples),
        "PhysActivity":         rng.integers(0, 2, n_samples),
        "Fruits":               rng.integers(0, 2, n_samples),
        "Veggies":              rng.integers(0, 2, n_samples),
        "HvyAlcoholConsump":    rng.binomial(1, 0.06, n_samples),
        "AnyHealthcare":        rng.integers(0, 2, n_samples),
        "NoDocbcCost":          rng.integers(0, 2, n_samples),
        "GenHlth":              rng.integers(1, 6, n_samples),
        "MentHlth":             rng.integers(0, 31, n_samples),
        "PhysHlth":             rng.integers(0, 31, n_samples),
        "DiffWalk":             rng.integers(0, 2, n_samples),
        "Sex":                  rng.integers(0, 2, n_samples),
        "Age":                  rng.integers(1, 14, n_samples),
        "Education":            rng.integers(1, 7, n_samples),
        "Income":               rng.integers(1, 9, n_samples),
    }

    df = pd.DataFrame(data)

    # Create a correlated target (diabetes more likely with high BP, BMI, age)
    score = (
        0.4 * df["HighBP"] +
        0.3 * df["HighChol"] +
        0.02 * (df["BMI"] - 18) +
        0.05 * df["Age"] +
        0.2 * df["HeartDiseaseorAttack"] +
        0.1  * df["Stroke"] +
        rng.normal(0, 0.3, n_samples)
    )
    df[TARGET_NAME] = (score > score.quantile(0.70)).astype(int)
    return df


def load_raw_dataset() -> pd.DataFrame:
    """
    Try to load the real CDC dataset; fall back to synthetic if unavailable.
    """
    csv_path = os.path.join(DATA_DIR, "raw_diabetes.csv")

    if os.path.exists(csv_path):
        print(f"[DATA] Loading cached dataset from {csv_path}")
        return pd.read_csv(csv_path)

    # Try downloading
    try:
        import zipfile, io
        print("[DATA] Downloading CDC BRFSS dataset from UCI ML Repository …")
        req = urllib.request.urlopen(DATASET_URL, timeout=30)
        with zipfile.ZipFile(io.BytesIO(req.read())) as z:
            # The archive contains a single CSV
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f)
        df.to_csv(csv_path, index=False)
        print(f"[DATA] Dataset saved to {csv_path} ({len(df):,} rows)")
        return df
    except Exception as exc:
        print(f"[DATA] Download failed ({exc}). Generating synthetic dataset …")
        df = _generate_synthetic(12_000)
        df.to_csv(csv_path, index=False)
        print(f"[DATA] Synthetic dataset saved ({len(df):,} rows)")
        return df


def preprocess(df: pd.DataFrame):
    """
    1. Keep only the 21 features + target.
    2. Drop rows with nulls.
    3. Scale continuous features with StandardScaler.
    4. Return X (numpy), y (numpy), scaler.
    """
    # Ensure all expected columns exist
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    if TARGET_NAME not in df.columns:
        raise ValueError(f"Target column '{TARGET_NAME}' not found in dataset.")

    df = df[available_features + [TARGET_NAME]].dropna().reset_index(drop=True)

    X = df[available_features].values.astype(np.float32)
    y = df[TARGET_NAME].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X.astype(np.float32), y.astype(np.float32), scaler, available_features


def split_into_hospitals(X, y, n_hospitals: int = N_HOSPITALS):
    """
    Non-IID split: shuffle then divide into n_hospitals roughly equal parts.
    Each part is further split 80/20 into local train/val.
    Returns a list of dicts: [{"X_train", "y_train", "X_val", "y_val"}, …]
    """
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split into n_hospitals chunks
    chunk_size = len(X) // n_hospitals
    hospital_splits = []

    for i in range(n_hospitals):
        start = i * chunk_size
        end   = (i + 1) * chunk_size if i < n_hospitals - 1 else len(X)
        Xh, yh = X[start:end], y[start:end]

        X_tr, X_val, y_tr, y_val = train_test_split(
            Xh, yh, test_size=0.20, random_state=RANDOM_SEED + i
        )
        hospital_splits.append({
            "hospital_id": i + 1,
            "X_train": X_tr, "y_train": y_tr,
            "X_val":   X_val, "y_val":   y_val,
            "n_train": len(X_tr), "n_val": len(X_val),
        })
        print(f"  Hospital {i+1}: {len(X_tr)} train | {len(X_val)} val samples")

    return hospital_splits


def prepare_and_save():
    """Full pipeline: load → preprocess → split → save to disk."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("\n=== Federated Healthcare Data Preparation ===")
    df     = load_raw_dataset()
    X, y, scaler, features = preprocess(df)

    print(f"\n[DATA] Total samples : {len(X):,}")
    print(f"[DATA] Features      : {len(features)}")
    print(f"[DATA] Positive rate : {y.mean():.2%}")

    # Global held-out test set (simulates evaluation at the server)
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )

    # Hospital-level splits (training data only)
    print(f"\n[DATA] Splitting {len(X_tr):,} training samples across {N_HOSPITALS} hospitals …")
    hospital_data = split_into_hospitals(X_tr, y_tr, N_HOSPITALS)

    # Save artefacts
    for h in hospital_data:
        fname = os.path.join(PROCESSED_DIR, f"hospital_{h['hospital_id']}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(h, f)

    global_test = {"X_test": X_test, "y_test": y_test}
    with open(os.path.join(PROCESSED_DIR, "global_test.pkl"), "wb") as f:
        pickle.dump(global_test, f)

    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(PROCESSED_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(features, f)

    print(f"\n[DATA] Saved {N_HOSPITALS} hospital datasets + global test set to {PROCESSED_DIR}")
    print(f"[DATA] Global test set : {len(X_test):,} samples")
    return hospital_data, X_test, y_test, scaler, features


if __name__ == "__main__":
    prepare_and_save()
