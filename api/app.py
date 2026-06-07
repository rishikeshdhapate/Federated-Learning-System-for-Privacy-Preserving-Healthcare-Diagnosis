"""
api/app.py
==========
Flask REST API exposing the federated learning system.

Endpoints
---------
GET  /            → Health check
POST /train       → Start / continue federated training (async thread)
GET  /status      → Current training status + latest metrics
POST /predict     → Predict disease risk from feature vector
GET  /history     → Full round-by-round training history
GET  /metrics     → Final model metrics
"""

import os
import sys
import json
import pickle
import threading
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.federated_server import FederatedServer

# ── Flask app setup ───────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow requests from the dashboard

# ── Configuration (override with environment variables) ─────────────────────

PROCESSED_DIR  = os.getenv("PROCESSED_DIR",  "data/processed")
OUTPUT_DIR     = os.getenv("OUTPUT_DIR",      "outputs")
N_HOSPITALS    = int(os.getenv("N_HOSPITALS",  "5"))
INPUT_DIM      = int(os.getenv("INPUT_DIM",    "21"))
N_ROUNDS       = int(os.getenv("N_ROUNDS",     "10"))
LOCAL_EPOCHS   = int(os.getenv("LOCAL_EPOCHS", "5"))
LR             = float(os.getenv("LR",         "0.001"))
DP_ENABLED     = os.getenv("DP_ENABLED",  "true").lower() == "true"
DP_NOISE_MULT  = float(os.getenv("DP_NOISE_MULT", "0.5"))
DP_MAX_NORM    = float(os.getenv("DP_MAX_NORM",   "1.0"))
DEVICE         = os.getenv("DEVICE", "cpu")

# ── Lazy-initialise the server (created once on first request) ─────────────

_server: FederatedServer = None
_server_lock = threading.Lock()
_train_thread = None


def _get_server() -> FederatedServer:
    global _server

    if _server is None:
        with _server_lock:
            if _server is None:

                # Ensure processed data exists
                test_file = os.path.join(PROCESSED_DIR, "global_test.pkl")

                if not os.path.exists(test_file):
                    print("[API] Processed data missing. Running preprocessing...")

                    from data.preprocess import prepare_and_save

                    prepare_and_save()

                _server = FederatedServer(
                    processed_dir=PROCESSED_DIR,
                    n_hospitals=N_HOSPITALS,
                    input_dim=INPUT_DIM,
                    n_rounds=N_ROUNDS,
                    local_epochs=LOCAL_EPOCHS,
                    lr=LR,
                    dp_enabled=DP_ENABLED,
                    dp_noise_mult=DP_NOISE_MULT,
                    dp_max_norm=DP_MAX_NORM,
                    output_dir=OUTPUT_DIR,
                    device=DEVICE,
                )

    return _server

def _load_scaler():
    """Load the feature scaler saved during pre-processing."""
    scaler_path = os.path.join(PROCESSED_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "service": "Federated Healthcare Learning API",
        "version": "1.0.0"
    })


@app.route("/train", methods=["POST"])
def start_training():
    """
    Trigger federated training in a background thread.
    Body (JSON, optional):
      {
        "n_rounds":     10,
        "local_epochs": 5,
        "dp_enabled":   true
      }
    """
    global _server, _train_thread

    # Parse optional overrides
    body = request.get_json(silent=True) or {}

    with _server_lock:
        srv = _get_server()

        # Update parameters if provided
        if "n_rounds" in body:
            srv.n_rounds = int(body["n_rounds"])
        if "local_epochs" in body:
            srv.local_epochs = int(body["local_epochs"])
            for c in srv.clients:
                c.local_epochs = srv.local_epochs
        if "dp_enabled" in body:
            dp = bool(body["dp_enabled"])
            for c in srv.clients:
                c.dp_enabled = dp

        if srv.is_training:
            return jsonify({"error": "Training already in progress"}), 409

        # Reset state for a fresh run
        srv.current_round    = 0
        srv.training_history = []
        srv.is_training      = True

    def _train():
        try:
            srv.run_all_rounds()
        except Exception as exc:
            srv.is_training    = False
            srv.status_message = f"Error: {exc}"
            print(f"[API] Training error: {exc}")

    _train_thread = threading.Thread(target=_train, daemon=True)
    _train_thread.start()

    return jsonify({
        "message":      "Federated training started",
        "n_rounds":     srv.n_rounds,
        "n_hospitals":  len(srv.clients),
        "dp_enabled":   srv.clients[0].dp_enabled if srv.clients else DP_ENABLED,
    }), 202


@app.route("/status", methods=["GET"])
def get_status():
    """Return the current training status and latest metrics."""
    srv = _get_server()
    status = srv.get_status()

    # Add history summary for the dashboard charts
    history_summary = []
    for entry in srv.training_history:
        gm = entry.get("global_metrics", {})
        history_summary.append({
            "round":    entry["round"],
            "accuracy": gm.get("accuracy", 0),
            "loss":     gm.get("loss", 0),
            "f1":       gm.get("f1", 0),
            "auc":      gm.get("auc", 0),
        })

    status["history_summary"] = history_summary
    return jsonify(status)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict disease risk for a single patient.

    Body (JSON):
    {
      "features": [f1, f2, … f21],   // raw (unscaled) feature values
      "feature_names": ["HighBP", …]  // optional, for documentation
    }
    """
    body = request.get_json(silent=True)
    if not body or "features" not in body:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    raw_features = body["features"]
    if len(raw_features) != INPUT_DIM:
        return jsonify({
            "error": f"Expected {INPUT_DIM} features, got {len(raw_features)}"
        }), 400

    # Scale features using the saved scaler
    scaler = _load_scaler()
    if scaler is not None:
        features_scaled = scaler.transform(
            np.array(raw_features, dtype=np.float32).reshape(1, -1)
        )[0].tolist()
    else:
        features_scaled = raw_features

    srv    = _get_server()
    result = srv.predict(features_scaled)
    result["input_features"] = raw_features

    return jsonify(result)


@app.route("/history", methods=["GET"])
def get_history():
    """Return the full round-by-round training history."""
    srv = _get_server()
    return jsonify({
        "n_rounds_completed": srv.current_round,
        "history":            srv.training_history,
    })


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Return the latest global model metrics."""
    srv = _get_server()
    if not srv.training_history:
        return jsonify({"error": "No training has been completed yet"}), 404

    last = srv.training_history[-1]
    return jsonify({
        "round":          last["round"],
        "global_metrics": last["global_metrics"],
        "client_metrics": last["client_metrics"],
        "duration_sec":   last["duration_sec"],
    })


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the server state (useful for re-training from scratch)."""
    global _server
    with _server_lock:
        _server = None
    return jsonify({"message": "Server state reset. Next request will re-initialise."})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Federated Healthcare Learning API")
    print("=" * 60)

    _get_server()

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )