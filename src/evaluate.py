"""
Evaluation: accuracy, precision, recall, F1, ROC/AUC, confusion matrix + plots.
Positive class = REAL (label 1).

Run:  python -m src.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR, REPORTS_DIR = ROOT / "models", ROOT / "reports"
CLASSES = ["FAKE", "REAL"]


def plot_training_curves(history: dict, path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(history["loss"]) + 1)
    ax[0].plot(ep, history["accuracy"], "o-", label="train")
    ax[0].plot(ep, history["val_accuracy"], "o-", label="validation")
    ax[0].set(title="Accuracy", xlabel="Epoch"); ax[0].legend()
    ax[1].plot(ep, history["loss"], "o-", label="train")
    ax[1].plot(ep, history["val_loss"], "o-", label="validation")
    ax[1].set(title="Loss", xlabel="Epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate(threshold: float = 0.5) -> dict:
    from tensorflow.keras.models import load_model

    REPORTS_DIR.mkdir(exist_ok=True)
    model = load_model(MODELS_DIR / "fake_news_model.h5", compile=False)
    data = np.load(MODELS_DIR / "test_data.npz")
    X, y = data["X_test"], data["y_test"]

    proba = model.predict(X, batch_size=256, verbose=0).ravel()
    pred = (proba >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y, proba)
    idx = np.linspace(0, len(fpr) - 1, min(500, len(fpr))).astype(int)  # keep JSON small
    auc = float(roc_auc_score(y, proba))
    cm = confusion_matrix(y, pred, labels=[0, 1])

    results = {
        "metrics": {
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred)),
            "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "roc_auc": auc,
        },
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "roc": {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(), "auc": auc},
        "classification_report": classification_report(y, pred, target_names=CLASSES,
                                                       output_dict=True),
        "n_test": int(len(y)),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    # ---- static PNGs for README / reports ---------------------------------
    hist_path = REPORTS_DIR / "history.json"
    if hist_path.exists():
        plot_training_curves(json.loads(hist_path.read_text()), REPORTS_DIR / "training_curves.png")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    print(json.dumps(results["metrics"], indent=2))
    return results


if __name__ == "__main__":
    evaluate()
