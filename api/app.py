"""
api/app.py
==========
Flask REST API exposing the federated learning system.
"""

import os
import sys
import json
import pickle
import threading
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Fix import path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

from server.federated_server import FederatedServer

# ── Flask app setup ───────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")
CORS(app)

# ── Configuration (FIXED PATHS) ──────────────────────────────────────────────

PROCESSED_DIR = os.getenv(
    "PROCESSED_DIR",
    os.path.join(BASE_DIR, "..", "data", "processed")
)

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    os.path.join(BASE_DIR, "..", "outputs")
)

N_HOSPITALS    = int(os.getenv("N_HOSPITALS",  "5"))
INPUT_DIM      = int(os.getenv("INPUT_DIM",    "21"))
N_ROUNDS       = int(os.getenv("N_ROUNDS",     "10"))
LOCAL_EPOCHS   = int(os.getenv("LOCAL_EPOCHS", "5"))
LR             = float(os.getenv("LR",         "0.001"))
DP_ENABLED     = os.getenv("DP_ENABLED",  "true").lower() == "true"
DP_NOISE_MULT  = float(os.getenv("DP_NOISE_MULT", "0.5"))
DP_MAX_NORM    = float(os.getenv("DP_MAX_NORM",   "1.0"))
DEVICE         = os.getenv("DEVICE", "cpu")

print(f"[DEBUG] PROCESSED_DIR: {PROCESSED_DIR}")
print(f"[DEBUG] OUTPUT_DIR: {OUTPUT_DIR}")

# ── Globals ─────────────────────────────────────────────────────────────────

_server: FederatedServer = None
_server_lock = threading.Lock()


def _get_server() -> FederatedServer:
    global _server
    if _server is None:
        with _server_lock:
            if _server is None:
                print("[DEBUG] Initializing FederatedServer...")
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
    scaler_path = os.path.join(PROCESSED_DIR, "scaler.pkl")
    print(f"[DEBUG] Loading scaler from: {scaler_path}")

    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load scaler: {e}")
            return None
    else:
        print("[WARNING] scaler.pkl not found")
    return None


# ── ROUTES ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def dashboard():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "running",
        "service": "Federated Healthcare Learning API",
        "version": "1.0.0"
    })


@app.route("/train", methods=["POST"])
def start_training():
    global _server

    body = request.get_json(silent=True) or {}

    with _server_lock:
        srv = _get_server()

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

        srv.current_round = 0
        srv.training_history = []
        srv.is_training = True

    # Run in background thread — returns immediately so Render's 30s request
    # timeout is never hit. Frontend polls /status to track progress.
    def _train_bg(server):
        try:
            print("[DEBUG] Background training started...")
            server.run_all_rounds()
            print("[DEBUG] Background training completed")
        except BaseException as exc:
            # Ensure is_training is always cleared even for non-Exception errors
            # (MemoryError, SystemExit, KeyboardInterrupt, etc.)
            server.is_training = False
            server.status_message = f"Error: {exc}"
            import traceback
            print(f"[ERROR] Training failed: {exc}")
            print(traceback.format_exc())

    t = threading.Thread(target=_train_bg, args=(srv,), daemon=True)
    t.start()

    return jsonify({
        "message": "Training started",
        "n_hospitals": srv.n_hospitals,
        "n_rounds": srv.n_rounds,
    }), 202


@app.route("/status", methods=["GET"])
def get_status():
    srv = _get_server()
    status = srv.get_status()

    history_summary = []
    for entry in srv.training_history:
        gm = entry.get("global_metrics", {})
        history_summary.append({
            "round": entry["round"],
            "accuracy": gm.get("accuracy", 0),
            "loss": gm.get("loss", 0),
            "f1": gm.get("f1", 0),
            "auc": gm.get("auc", 0),
        })

    status["history_summary"] = history_summary
    return jsonify(status)


@app.route("/predict", methods=["POST"])
def predict():
    HIDDEN_FEATURE_DEFAULTS = {
        "Fruits": 1.0,
        "Veggies": 1.0,
        "NoDocbcCost": 0.0,
        "PhysHlth": 0.0,
        "Education": 4.0,
        "Income": 5.0,
    }

    DASHBOARD_FEATURE_NAMES = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
        "Stroke", "HeartDiseaseorAttack", "PhysActivity",
        "HvyAlcoholConsump", "AnyHealthcare", "GenHlth",
        "MentHlth", "DiffWalk", "Sex", "Age",
    ]

    FULL_FEATURE_NAMES = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
        "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
        "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
        "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
        "Sex", "Age", "Education", "Income",
    ]

    body = request.get_json(silent=True)
    if not body or "features" not in body:
        return jsonify({"error": "Missing features"}), 400

    raw_features = body["features"]

    if len(raw_features) == len(DASHBOARD_FEATURE_NAMES):
        dashboard_map = dict(zip(DASHBOARD_FEATURE_NAMES, raw_features))
        raw_features = [
            dashboard_map.get(name, HIDDEN_FEATURE_DEFAULTS.get(name, 0.0))
            for name in FULL_FEATURE_NAMES
        ]

    scaler = _load_scaler()
    if scaler is not None:
        features_scaled = scaler.transform(
            np.array(raw_features, dtype=np.float32).reshape(1, -1)
        )[0].tolist()
    else:
        features_scaled = raw_features

    srv = _get_server()

    try:
        result = srv.predict(features_scaled)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

    result["input_features"] = raw_features
    return jsonify(result)


@app.route("/history", methods=["GET"])
def get_history():
    srv = _get_server()
    return jsonify({"history": srv.training_history})


@app.route("/metrics", methods=["GET"])
def get_metrics():
    srv = _get_server()
    if not srv.training_history:
        return jsonify({"error": "No training yet"}), 404

    return jsonify(srv.training_history[-1])


@app.route("/reset", methods=["POST"])
def reset():
    global _server
    with _server_lock:
        _server = None
    return jsonify({"message": "reset done"})


# ── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Federated Healthcare API...")

    _get_server()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

"""
api/app.py
==========
Flask REST API exposing the federated learning system.
"""

import os
import sys
import json
import pickle
import threading
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Fix import path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

from server.federated_server import FederatedServer

# ── Flask app setup ───────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")
CORS(app)

# ── Configuration (FIXED PATHS) ──────────────────────────────────────────────

PROCESSED_DIR = os.getenv(
    "PROCESSED_DIR",
    os.path.join(BASE_DIR, "..", "data", "processed")
)

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    os.path.join(BASE_DIR, "..", "outputs")
)

N_HOSPITALS    = int(os.getenv("N_HOSPITALS",  "5"))
INPUT_DIM      = int(os.getenv("INPUT_DIM",    "21"))
N_ROUNDS       = int(os.getenv("N_ROUNDS",     "10"))
LOCAL_EPOCHS   = int(os.getenv("LOCAL_EPOCHS", "5"))
LR             = float(os.getenv("LR",         "0.001"))
DP_ENABLED     = os.getenv("DP_ENABLED",  "true").lower() == "true"
DP_NOISE_MULT  = float(os.getenv("DP_NOISE_MULT", "0.5"))
DP_MAX_NORM    = float(os.getenv("DP_MAX_NORM",   "1.0"))
DEVICE         = os.getenv("DEVICE", "cpu")

print(f"[DEBUG] PROCESSED_DIR: {PROCESSED_DIR}")
print(f"[DEBUG] OUTPUT_DIR: {OUTPUT_DIR}")

# ── Globals ─────────────────────────────────────────────────────────────────

_server: FederatedServer = None
_server_lock = threading.Lock()


def _get_server() -> FederatedServer:
    global _server
    if _server is None:
        with _server_lock:
            if _server is None:
                print("[DEBUG] Initializing FederatedServer...")
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
    scaler_path = os.path.join(PROCESSED_DIR, "scaler.pkl")
    print(f"[DEBUG] Loading scaler from: {scaler_path}")

    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load scaler: {e}")
            return None
    else:
        print("[WARNING] scaler.pkl not found")
    return None


# ── ROUTES ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def dashboard():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "running",
        "service": "Federated Healthcare Learning API",
        "version": "1.0.0"
    })


@app.route("/train", methods=["POST"])
def start_training():
    global _server

    body = request.get_json(silent=True) or {}

    with _server_lock:
        srv = _get_server()

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

        srv.current_round = 0
        srv.training_history = []
        srv.is_training = True

    # Run in background thread — returns immediately so Render's 30s request
    # timeout is never hit. Frontend polls /status to track progress.
    def _train_bg(server):
        try:
            print("[DEBUG] Background training started...")
            server.run_all_rounds()
            print("[DEBUG] Background training completed")
        except Exception as exc:
            server.is_training = False
            server.status_message = f"Error: {exc}"
            print(f"[ERROR] Training failed: {exc}")

    t = threading.Thread(target=_train_bg, args=(srv,), daemon=True)
    t.start()

    return jsonify({
        "message": "Training started",
        "n_hospitals": srv.n_hospitals,
        "n_rounds": srv.n_rounds,
    }), 202


@app.route("/status", methods=["GET"])
def get_status():
    srv = _get_server()
    status = srv.get_status()

    history_summary = []
    for entry in srv.training_history:
        gm = entry.get("global_metrics", {})
        history_summary.append({
            "round": entry["round"],
            "accuracy": gm.get("accuracy", 0),
            "loss": gm.get("loss", 0),
            "f1": gm.get("f1", 0),
            "auc": gm.get("auc", 0),
        })

    status["history_summary"] = history_summary
    return jsonify(status)


@app.route("/predict", methods=["POST"])
def predict():
    HIDDEN_FEATURE_DEFAULTS = {
        "Fruits": 1.0,
        "Veggies": 1.0,
        "NoDocbcCost": 0.0,
        "PhysHlth": 0.0,
        "Education": 4.0,
        "Income": 5.0,
    }

    DASHBOARD_FEATURE_NAMES = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
        "Stroke", "HeartDiseaseorAttack", "PhysActivity",
        "HvyAlcoholConsump", "AnyHealthcare", "GenHlth",
        "MentHlth", "DiffWalk", "Sex", "Age",
    ]

    FULL_FEATURE_NAMES = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
        "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
        "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
        "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
        "Sex", "Age", "Education", "Income",
    ]

    body = request.get_json(silent=True)
    if not body or "features" not in body:
        return jsonify({"error": "Missing features"}), 400

    raw_features = body["features"]

    if len(raw_features) == len(DASHBOARD_FEATURE_NAMES):
        dashboard_map = dict(zip(DASHBOARD_FEATURE_NAMES, raw_features))
        raw_features = [
            dashboard_map.get(name, HIDDEN_FEATURE_DEFAULTS.get(name, 0.0))
            for name in FULL_FEATURE_NAMES
        ]

    scaler = _load_scaler()
    if scaler is not None:
        features_scaled = scaler.transform(
            np.array(raw_features, dtype=np.float32).reshape(1, -1)
        )[0].tolist()
    else:
        features_scaled = raw_features

    srv = _get_server()

    try:
        result = srv.predict(features_scaled)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

    result["input_features"] = raw_features
    return jsonify(result)


@app.route("/history", methods=["GET"])
def get_history():
    srv = _get_server()
    return jsonify({"history": srv.training_history})


@app.route("/metrics", methods=["GET"])
def get_metrics():
    srv = _get_server()
    if not srv.training_history:
        return jsonify({"error": "No training yet"}), 404

    return jsonify(srv.training_history[-1])


@app.route("/reset", methods=["POST"])
def reset():
    global _server
    with _server_lock:
        _server = None
    return jsonify({"message": "reset done"})


# ── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Federated Healthcare API...")

    _get_server()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False) 
