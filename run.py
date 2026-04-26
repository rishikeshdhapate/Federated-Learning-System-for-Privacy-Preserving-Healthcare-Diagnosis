"""
run.py
======
End-to-end runner for the Federated Healthcare Learning System.
Runs without Docker — ideal for local development and testing.

Usage
-----
    python run.py                          # default settings
    python run.py --rounds 15 --epochs 3  # custom settings
    python run.py --no-dp                 # disable differential privacy
    python run.py --api-only              # start API only (already trained)
"""

import argparse
import os
import sys
import threading

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="FedHealth — Federated Learning Runner")
parser.add_argument("--rounds",     type=int,   default=10,    help="Federated rounds")
parser.add_argument("--epochs",     type=int,   default=5,     help="Local epochs per round")
parser.add_argument("--hospitals",  type=int,   default=5,     help="Number of hospital nodes")
parser.add_argument("--lr",         type=float, default=0.001, help="Learning rate")
parser.add_argument("--no-dp",      action="store_true",       help="Disable differential privacy")
parser.add_argument("--api-only",   action="store_true",       help="Skip training, launch API only")
parser.add_argument("--api-port",   type=int,   default=5000,  help="Flask API port")
parser.add_argument("--device",     type=str,   default="cpu", help="cpu or cuda")
args = parser.parse_args()

DP_ENABLED = not args.no_dp

print("=" * 65)
print("  🏥  FedHealth — Privacy-Preserving Federated Diagnosis")
print("=" * 65)
print(f"  Rounds      : {args.rounds}")
print(f"  Hospitals   : {args.hospitals}")
print(f"  Local epochs: {args.epochs}")
print(f"  Learning rate: {args.lr}")
print(f"  Diff. Privacy: {'✓ ENABLED' if DP_ENABLED else '✗ DISABLED'}")
print(f"  Device       : {args.device}")
print("=" * 65)

# ── Step 1: Data preparation ──────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"

if not os.path.exists(os.path.join(PROCESSED_DIR, "global_test.pkl")):
    print("\n[STEP 1] Preparing dataset …")
    import importlib.util, types
    # Run the preprocess script
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    # Update N_HOSPITALS before importing
    import data.preprocess as prep
    prep.N_HOSPITALS = args.hospitals
    prep.prepare_and_save()
else:
    print("\n[STEP 1] Dataset already prepared — skipping.")

# ── Step 2: Federated training ────────────────────────────────────────────────

if not args.api_only:
    print("\n[STEP 2] Starting federated training …\n")
    from server.federated_server import FederatedServer

    server = FederatedServer(
        processed_dir = PROCESSED_DIR,
        n_hospitals   = args.hospitals,
        input_dim     = 21,
        n_rounds      = args.rounds,
        local_epochs  = args.epochs,
        lr            = args.lr,
        dp_enabled    = DP_ENABLED,
        dp_noise_mult = 0.5,
        dp_max_norm   = 1.0,
        output_dir    = "outputs",
        device        = args.device,
    )
    result = server.run_all_rounds()

    # Generate visualisations
    print("\n[STEP 3] Generating visualisation plots …")
    try:
        sys.path.insert(0, "outputs")
        from outputs import visualize
        import json
        # The history was saved by the server; load and plot
        hist_path = "outputs/training_history.json"
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
            visualize.plot_training_curves(history)
            visualize.plot_per_hospital(history)
            visualize.plot_confusion_matrix(history)
            visualize.print_summary(history)
    except Exception as e:
        print(f"[WARN] Visualisation failed: {e}")

else:
    print("\n[STEP 2] Skipping training (--api-only flag set).")
    server = None

# ── Step 3: Start Flask API ───────────────────────────────────────────────────

print(f"\n[STEP 4] Launching Flask API on port {args.api_port} …")
print(f"         Dashboard: open dashboard/index.html in your browser")
print(f"         API docs : http://localhost:{args.api_port}/")
print()

# Set env vars for the API module
os.environ["N_HOSPITALS"]   = str(args.hospitals)
os.environ["N_ROUNDS"]      = str(args.rounds)
os.environ["LOCAL_EPOCHS"]  = str(args.epochs)
os.environ["LR"]            = str(args.lr)
os.environ["DP_ENABLED"]    = "true" if DP_ENABLED else "false"
os.environ["PROCESSED_DIR"] = PROCESSED_DIR
os.environ["DEVICE"]        = args.device

from api.app import app, _get_server

# Pre-warm the server singleton so it loads the saved model
if not args.api_only and server is not None:
    # Inject already-trained server instance to avoid re-initialising
    import api.app as api_module
    api_module._server = server

app.run(host="0.0.0.0", port=args.api_port, debug=False, threaded=True)
