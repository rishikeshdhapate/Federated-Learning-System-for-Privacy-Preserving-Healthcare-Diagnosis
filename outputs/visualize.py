"""
outputs/visualize.py
====================
Generates and saves training visualisation plots from training_history.json.
Run this after training completes:

    python outputs/visualize.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "training_history.json")
OUT_DIR      = os.path.dirname(__file__)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#050d1a",
    "axes.facecolor":   "#0a1628",
    "axes.edgecolor":   "#1a3050",
    "axes.labelcolor":  "#5a7a9a",
    "xtick.color":      "#5a7a9a",
    "ytick.color":      "#5a7a9a",
    "grid.color":       "#1a3050",
    "text.color":       "#e2eaf5",
    "font.family":      "monospace",
})
ACCENT  = "#00d4ff"
ACCENT2 = "#7c3aed"
GREEN   = "#10b981"
DANGER  = "#f43f5e"
WARN    = "#f59e0b"


def load_history():
    if not os.path.exists(HISTORY_FILE):
        print(f"[VIZ] History file not found: {HISTORY_FILE}")
        print("[VIZ] Run training first (POST /train)")
        return None
    with open(HISTORY_FILE) as f:
        return json.load(f)


def plot_training_curves(history):
    """Accuracy, loss, F1 and AUC curves across rounds."""
    rounds     = [e["round"]                          for e in history]
    accuracies = [e["global_metrics"]["accuracy"]     for e in history]
    losses     = [e["global_metrics"]["loss"]         for e in history]
    f1s        = [e["global_metrics"]["f1"]           for e in history]
    aucs       = [e["global_metrics"]["auc"]          for e in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Federated Learning Training Curves", color="#e2eaf5",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.patch.set_facecolor("#050d1a")

    def plot_one(ax, x, y, label, color, ylabel):
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5,
                markerfacecolor=color, markeredgewidth=0)
        ax.fill_between(x, y, alpha=0.12, color=color)
        ax.set_title(label, color=color, fontsize=12, pad=8)
        ax.set_xlabel("Round", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.set_xlim(min(x) - 0.3, max(x) + 0.3)

    plot_one(axes[0,0], rounds, accuracies, "Global Accuracy", ACCENT, "Accuracy")
    plot_one(axes[0,1], rounds, losses,     "Global Loss",     ACCENT2,"Loss")
    plot_one(axes[1,0], rounds, f1s,        "F1 Score",        GREEN,  "F1")
    plot_one(axes[1,1], rounds, aucs,       "AUC-ROC",         WARN,   "AUC")

    for ax in axes.flat:
        ax.set_facecolor("#0a1628")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a3050")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#050d1a")
    print(f"[VIZ] Saved: {out_path}")
    plt.close()


def plot_per_hospital(history):
    """Per-hospital accuracy across rounds."""
    last = history[-1]
    clients = last.get("client_metrics", [])
    if not clients:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#050d1a")
    ax.set_facecolor("#0a1628")

    colors = [ACCENT, ACCENT2, GREEN, WARN, DANGER]

    for entry in history:
        round_n = entry["round"]
        for cm in entry.get("client_metrics", []):
            hid = cm["hospital_id"]
            ax.scatter(round_n, cm["accuracy"],
                       color=colors[(hid-1) % len(colors)],
                       s=50, zorder=3, label=f"Hospital {hid}" if round_n == 1 else "")

    # Draw per-hospital lines
    for hid in range(1, len(clients)+1):
        xs = [e["round"] for e in history]
        ys = []
        for e in history:
            cml = [c for c in e.get("client_metrics",[]) if c["hospital_id"]==hid]
            ys.append(cml[0]["accuracy"] if cml else None)
        valid = [(x,y) for x,y in zip(xs,ys) if y is not None]
        if valid:
            vx, vy = zip(*valid)
            ax.plot(vx, vy, color=colors[(hid-1)%len(colors)], linewidth=1.5, alpha=0.7)

    # Global model line
    gx = [e["round"] for e in history]
    gy = [e["global_metrics"]["accuracy"] for e in history]
    ax.plot(gx, gy, color="white", linewidth=2.5, linestyle="--",
            label="Global Model", zorder=5)

    ax.set_title("Per-Hospital vs Global Model Accuracy", color="#e2eaf5", fontsize=13)
    ax.set_xlabel("Round", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.legend(loc="lower right", framealpha=0.2, labelcolor="#e2eaf5")
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a3050")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "per_hospital_accuracy.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#050d1a")
    print(f"[VIZ] Saved: {out_path}")
    plt.close()


def plot_confusion_matrix(history):
    """Confusion matrix heatmap from the last round."""
    last = history[-1]
    cm_raw = last["global_metrics"].get("confusion_matrix", [[0,0],[0,0]])
    cm = np.array(cm_raw)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#050d1a")
    ax.set_facecolor("#0a1628")

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)

    classes = ["Low Risk (0)", "High Risk (1)"]
    ax.set_xticks([0,1]); ax.set_xticklabels(classes)
    ax.set_yticks([0,1]); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title(f"Confusion Matrix — Round {last['round']}", color="#e2eaf5", fontsize=12)

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f"{cm[i,j]}\n({100*cm[i,j]/max(total,1):.1f}%)",
                           ha="center", va="center",
                           color="white" if cm[i,j] < cm.max()*0.6 else "black",
                           fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#050d1a")
    print(f"[VIZ] Saved: {out_path}")
    plt.close()


def print_summary(history):
    last = history[-1]
    gm   = last["global_metrics"]
    print("\n" + "="*55)
    print("  FEDERATED TRAINING SUMMARY")
    print("="*55)
    print(f"  Rounds completed : {last['round']}")
    print(f"  Final Accuracy   : {gm['accuracy']:.4f}")
    print(f"  Final Precision  : {gm['precision']:.4f}")
    print(f"  Final Recall     : {gm['recall']:.4f}")
    print(f"  Final F1         : {gm['f1']:.4f}")
    print(f"  Final AUC-ROC    : {gm['auc']:.4f}")
    print(f"  Final Loss       : {gm['loss']:.6f}")
    print("="*55)


if __name__ == "__main__":
    history = load_history()
    if history:
        print(f"[VIZ] Generating plots from {len(history)} training rounds …")
        plot_training_curves(history)
        plot_per_hospital(history)
        plot_confusion_matrix(history)
        print_summary(history)
        print("\n[VIZ] All plots saved to outputs/")
