"""
Inference + prediction history.

    from src.predict import FakeNewsPredictor
    FakeNewsPredictor().predict("Some article ...")
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocess import texts_to_padded

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR, REPORTS_DIR = ROOT / "models", ROOT / "reports"
HISTORY_PATH = REPORTS_DIR / "prediction_history.csv"
REQUIRED = ["fake_news_model.h5", "tokenizer.pkl", "label_encoder.pkl", "config.json"]


def models_ready() -> bool:
    return all((MODELS_DIR / f).exists() for f in REQUIRED)


class FakeNewsPredictor:
    """Wraps model + tokenizer + label encoder. Model output = P(REAL)."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        from tensorflow.keras.models import load_model  # lazy: keeps imports light
        self.model = load_model(Path(models_dir) / "fake_news_model.h5", compile=False)
        self.tokenizer = joblib.load(Path(models_dir) / "tokenizer.pkl")
        self.label_encoder = joblib.load(Path(models_dir) / "label_encoder.pkl")
        cfg = json.loads((Path(models_dir) / "config.json").read_text())
        self.max_len, self.threshold = cfg["max_len"], cfg.get("threshold", 0.5)

    # -- batch APIs used by SHAP / LIME -------------------------------------
    def predict_proba(self, texts) -> np.ndarray:
        """P(REAL) for each raw text, shape (n,)."""
        texts = [str(t) for t in texts]
        X = texts_to_padded(self.tokenizer, texts, self.max_len)
        return self.model.predict(X, batch_size=64, verbose=0).ravel()

    def predict_proba_two(self, texts) -> np.ndarray:
        """[P(FAKE), P(REAL)] columns — the format LIME expects."""
        p = self.predict_proba(texts)
        return np.column_stack([1 - p, p])

    # -- single-article API --------------------------------------------------
    def predict(self, text: str) -> dict:
        p_real = float(self.predict_proba([text])[0])
        idx = int(p_real >= self.threshold)
        label = str(self.label_encoder.inverse_transform([idx])[0])
        return {
            "label": label,
            "confidence": p_real if idx == 1 else 1 - p_real,
            "prob_real": p_real,
            "prob_fake": 1 - p_real,
        }


# ------------------------------- history ------------------------------------
def save_prediction(text: str, result: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "excerpt": " ".join(text.split())[:140],
        "prediction": result["label"],
        "confidence": round(result["confidence"], 4),
        "prob_real": round(result["prob_real"], 4),
    }])
    row.to_csv(HISTORY_PATH, mode="a", header=not HISTORY_PATH.exists(), index=False)


def load_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH).iloc[::-1].reset_index(drop=True)  # newest first
    return pd.DataFrame(columns=["timestamp", "excerpt", "prediction", "confidence", "prob_real"])


def clear_history() -> None:
    HISTORY_PATH.unlink(missing_ok=True)
