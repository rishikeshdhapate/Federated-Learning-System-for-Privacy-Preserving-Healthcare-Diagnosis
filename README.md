# 🏥 FedHealth — Privacy-Preserving Federated Learning for Healthcare Diagnosis

> A complete end-to-end Federated Learning system where multiple hospitals collaboratively train a disease prediction model **without sharing raw patient data**.

---

## 📋 Table of Contents

1. [What Is Federated Learning?](#what-is-federated-learning)
2. [System Architecture](#system-architecture)
3. [Privacy Features](#privacy-features)
4. [Project Structure](#project-structure)
5. [Quick Start (Local)](#quick-start-local)
6. [Docker Deployment](#docker-deployment)
7. [API Reference](#api-reference)
8. [Dashboard](#dashboard)
9. [Dataset](#dataset)
10. [Model Details](#model-details)
11. [Results & Metrics](#results--metrics)
12. [Configuration](#configuration)

---

## What Is Federated Learning?

Traditional machine learning requires centralising all data in one place — a serious problem in healthcare where patient data is sensitive and legally protected (HIPAA, GDPR).

**Federated Learning** solves this:

```
Hospital A  →  train locally  →  send only model weights
Hospital B  →  train locally  →  send only model weights   →  FedAvg  →  Global Model
Hospital C  →  train locally  →  send only model weights
```

Raw data **never leaves** the hospital. Only model weight updates are shared.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FEDERATED SERVER                          │
│  ┌─────────────┐   FedAvg    ┌─────────────────────────┐  │
│  │ Global Model│◄────────────│   Aggregation Engine     │  │
│  │  (PyTorch)  │────────────►│  Secure Aggregation Sim  │  │
│  └─────────────┘             └─────────────────────────┘  │
│         ▲                              ▲                     │
│         │ weights only                │ masked updates       │
└─────────┼──────────────────────────────────────────────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│Hosp. 1 │  │Hosp. 2 │  │Hosp. 3 │  │Hosp. 4 │  │Hosp. 5 │
│Local   │  │Local   │  │Local   │  │Local   │  │Local   │
│Data ✗  │  │Data ✗  │  │Data ✗  │  │Data ✗  │  │Data ✗  │
│DP-SGD  │  │DP-SGD  │  │DP-SGD  │  │DP-SGD  │  │DP-SGD  │
└────────┘  └────────┘  └────────┘  └────────┘  └────────┘
```

**Communication flow (one round):**
1. Server sends global model weights to all hospitals
2. Each hospital trains on its private data (with DP noise)
3. Hospitals return masked weight updates (secure aggregation)
4. Server runs FedAvg → new global model
5. Repeat for N rounds

---

## Privacy Features

| Feature | Implementation | File |
|---------|---------------|------|
| **No raw data sharing** | Data stays in `data/processed/hospital_*.pkl` | `client/local_trainer.py` |
| **Differential Privacy** | Gradient clipping + Gaussian noise (DP-SGD) | `client/local_trainer.py` |
| **Secure Aggregation** | Pairwise mask simulation (cancel in sum) | `server/federated_server.py` |
| **Weight-only exchange** | Only `state_dict` tensors transmitted | Both |

### Differential Privacy Details

```
After each backward pass:
1. Clip gradients: ||g|| ≤ C  (sensitivity bound, C = dp_max_norm)
2. Add noise: g ← g + N(0, σ²I)  where σ = noise_mult × C / batch_size
```

This provides (ε, δ)-differential privacy. For production, integrate [Opacus](https://opacus.ai/) for formal DP accounting.

---

## Project Structure

```
federated-health/
│
├── model/                       # Neural network & metrics
│   ├── neural_network.py        # MLP architecture (4-layer)
│   └── metrics.py               # Evaluation functions
│
├── client/                      # Hospital client simulation
│   └── local_trainer.py         # DP-SGD local training
│
├── server/                      # Central federated server
│   └── federated_server.py      # FedAvg + secure aggregation
│
├── api/                         # REST API
│   └── app.py                   # Flask endpoints
│
├── dashboard/                   # Web dashboard
│   └── index.html               # Live monitoring UI
│
├── data/                        # Data pipeline
│   ├── preprocess.py            # CDC dataset download & split
│   └── processed/               # Hospital data splits (auto-generated)
│
├── outputs/                     # Model outputs
│   ├── visualize.py             # Plot generation script
│   ├── global_model.pt          # Final trained model
│   └── training_history.json    # Round-by-round metrics
│
├── docker/                      # Docker configuration
│   ├── Dockerfile.server        # Server + API container
│   ├── Dockerfile.client        # Hospital client container
│   └── Dockerfile.dashboard     # Dashboard nginx container
│
├── docker-compose.yml           # Full system orchestration
├── requirements.txt             # Python dependencies
├── run.py                       # End-to-end runner script
└── README.md                    # This file
```

---

## Quick Start (Local)

### Prerequisites

- Python 3.9 or 3.10
- pip

### Step 1 — Clone and install dependencies

```bash
git clone <your-repo-url>
cd federated-health
pip install -r requirements.txt
```

### Step 2 — Run the complete system

```bash
python run.py
```

This will:
1. Download (or generate) the CDC BRFSS diabetes dataset
2. Split it into 5 hospital data silos
3. Run 10 rounds of federated training with differential privacy
4. Generate visualisation plots in `outputs/`
5. Launch the Flask API on `http://localhost:5000`

Then open `dashboard/index.html` in your browser.

### Step 3 — Try the API

```bash
# Check API health
curl http://localhost:5000/

# Start training via API
curl -X POST http://localhost:5000/train \
     -H "Content-Type: application/json" \
     -d '{"n_rounds": 10, "local_epochs": 5, "dp_enabled": true}'

# Check training status
curl http://localhost:5000/status

# Get a prediction
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [1,1,1,35,0,0,0,1,1,1,0,1,0,3,5,5,0,0,8,5,6]}'
```

### Custom options

```bash
# More rounds, no differential privacy
python run.py --rounds 20 --epochs 3 --no-dp

# Start API only (model already trained)
python run.py --api-only

# Use GPU if available
python run.py --device cuda
```

---

## Docker Deployment

### Prerequisites

- Docker Engine 24+
- Docker Compose v2

### Build and launch all services

```bash
# Build images and start everything
docker-compose up --build

# Or in detached mode
docker-compose up --build -d
```

### Access the system

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:8080 |
| **API** | http://localhost:5000 |

### Useful Docker commands

```bash
# View server logs
docker-compose logs -f server

# View a specific hospital's logs
docker-compose logs -f hospital_1

# Stop everything
docker-compose down

# Remove all volumes and rebuild
docker-compose down -v && docker-compose up --build
```

---

## API Reference

### `GET /`
Health check.
```json
{"status": "running", "service": "Federated Healthcare Learning API"}
```

### `POST /train`
Start federated training. Returns immediately; training runs in background.

**Request body (optional):**
```json
{
  "n_rounds":     10,
  "local_epochs": 5,
  "dp_enabled":   true
}
```

**Response (202):**
```json
{
  "message": "Federated training started",
  "n_rounds": 10,
  "n_hospitals": 5,
  "dp_enabled": true
}
```

### `GET /status`
Poll training progress.
```json
{
  "is_training": true,
  "current_round": 4,
  "total_rounds": 10,
  "status_message": "Round 4/10 in progress …",
  "last_metrics": {
    "accuracy": 0.7823,
    "f1": 0.7145,
    "auc": 0.8234,
    "loss": 0.4521
  },
  "history_summary": [
    {"round": 1, "accuracy": 0.71, "loss": 0.56, "f1": 0.68, "auc": 0.78},
    ...
  ]
}
```

### `POST /predict`
Predict disease risk for a single patient.

**Request body:**
```json
{
  "features": [1, 1, 1, 35, 0, 0, 0, 1, 1, 1, 0, 1, 0, 3, 5, 5, 0, 0, 8, 5, 6]
}
```
Features (21 values in order):
`HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income`

**Response:**
```json
{
  "probability": 0.7823,
  "prediction": 1,
  "label": "High Risk",
  "confidence": 56.5
}
```

### `GET /metrics`
Latest global model metrics.

### `GET /history`
Full round-by-round training history.

### `POST /reset`
Reset the server state for a fresh training run.

---

## Dashboard

Open `dashboard/index.html` in your browser (or access via Docker at http://localhost:8080).

### Features

- **Live accuracy & loss curves** — updated every 2.5 seconds during training
- **Hospital node status** — shows per-hospital accuracy and DP status
- **Confusion matrix** — visualises true/false positives and negatives
- **Training controls** — adjust rounds, epochs, and DP toggle
- **Patient prediction** — enter feature values or use preset profiles
- **Activity log** — real-time training event log

---

## Dataset

**CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015**
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/cdc+diabetes+health+indicators)
- ~253,000 survey responses
- Binary classification: diabetes risk (0 = no, 1 = yes)
- 21 health features (demographics, lifestyle, medical history)

If the download fails in an offline environment, a realistic synthetic dataset is automatically generated with the same schema and statistical properties.

### Data splits

The dataset is divided into 5 non-overlapping parts (simulating different hospital populations):

| Hospital | Training samples | Validation samples |
|----------|-----------------|-------------------|
| 1 | ~16,000 | ~4,000 |
| 2 | ~16,000 | ~4,000 |
| 3 | ~16,000 | ~4,000 |
| 4 | ~16,000 | ~4,000 |
| 5 | ~16,000 | ~4,000 |

Global test set: ~40,000 samples (held out at the server, never used for training)

---

## Model Details

**Architecture:** Multi-Layer Perceptron (MLP)

```
Input (21) → FC(64) → BN → ReLU → Dropout(0.3)
           → FC(128) → BN → ReLU → Dropout(0.3)
           → FC(64) → BN → ReLU → Dropout(0.3)
           → FC(32) → ReLU
           → FC(1) → Sigmoid
```

- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam (lr=1e-3)
- **DP-SGD:** Gradient clipping (C=1.0) + Gaussian noise (σ=0.5)
- **Aggregation:** Weighted FedAvg (by number of local samples)

---

## Results & Metrics

Typical results after 10 federated rounds (may vary):

| Metric | Value |
|--------|-------|
| Accuracy | ~78–82% |
| Precision | ~74–79% |
| Recall | ~72–78% |
| F1 Score | ~73–78% |
| AUC-ROC | ~83–87% |

Generate plots after training:
```bash
python outputs/visualize.py
```

Outputs:
- `outputs/training_curves.png` — accuracy, loss, F1, AUC per round
- `outputs/per_hospital_accuracy.png` — per-hospital vs global model
- `outputs/confusion_matrix.png` — confusion matrix heatmap
- `outputs/global_model.pt` — final PyTorch model weights

---

## Configuration

All parameters can be set via environment variables (for Docker) or CLI flags (for `run.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_HOSPITALS` | 5 | Number of hospital clients |
| `N_ROUNDS` | 10 | Federated communication rounds |
| `LOCAL_EPOCHS` | 5 | Local epochs per round per client |
| `LR` | 0.001 | Local learning rate |
| `DP_ENABLED` | true | Enable differential privacy |
| `DP_NOISE_MULT` | 0.5 | DP noise multiplier (σ/C) |
| `DP_MAX_NORM` | 1.0 | Gradient clipping norm (C) |
| `DEVICE` | cpu | PyTorch device (cpu / cuda) |

---

## For Final Year Submission

This project demonstrates:

1. **Federated Learning** — FedAvg algorithm with multiple hospital nodes
2. **Differential Privacy** — DP-SGD with gradient clipping and Gaussian noise
3. **Secure Aggregation** — Pairwise mask simulation
4. **Healthcare ML** — Disease risk prediction on real public dataset
5. **System Design** — Microservices with Docker, REST API, live dashboard
6. **Software Engineering** — Clean code structure, documentation, error handling

### Extensions for extra marks
- Replace synthetic secure aggregation with real SMPC (use PySyft)
- Add formal DP accounting with Opacus (ε, δ guarantees)
- Use CNN on medical imaging datasets (chest X-rays, retinal scans)
- Add authentication to the API
- Deploy on AWS/GCP with real network separation between hospitals

---

## License

MIT — free to use for educational purposes.
