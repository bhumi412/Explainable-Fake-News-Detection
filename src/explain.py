"""
Explainable AI: SHAP + LIME for the BiLSTM classifier.

Sign convention (both methods explain the model's P(REAL) output):
    positive value -> pushes the prediction toward REAL
    negative value -> pushes the prediction toward FAKE

CLI:  python -m src.explain --text "Some article ..."   # saves PNGs to reports/
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
GREEN, RED = "#2ECC71", "#E74C3C"


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


# ------------------------------- SHAP ---------------------------------------
def explain_shap(predictor, text: str, max_words: int = 120, max_evals: int = 600):
    """
    Model-agnostic SHAP (Partition explainer with a text masker). Words are masked
    and the model is re-queried, so it works with any Keras/TensorFlow version.

    Returns (DataFrame[token, value], base_value). base_value = model output when
    every word is masked (the 'average' article).
    """
    import shap

    text = _truncate(text, max_words)
    n_tokens = len(re.split(r"\W+", text))
    max_evals = max(max_evals, 2 * n_tokens + 10)

    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(lambda xs: predictor.predict_proba(list(xs)), masker)
    sv = explainer([text], max_evals=max_evals, batch_size=32)

    tokens = list(sv.data[0])
    values = np.asarray(sv.values[0]).reshape(len(tokens), -1)[:, 0]
    base = float(np.asarray(sv.base_values[0]).ravel()[0])
    return pd.DataFrame({"token": tokens, "value": values}), base


# ------------------------------- LIME ---------------------------------------
def explain_lime(predictor, text: str, num_features: int = 15,
                 num_samples: int = 500, max_words: int = 200):
    """Returns (list[(word, weight)], lime Explanation object) for class REAL."""
    from lime.lime_text import LimeTextExplainer

    text = _truncate(text, max_words)
    explainer = LimeTextExplainer(class_names=["FAKE", "REAL"], random_state=42)
    exp = explainer.explain_instance(text, predictor.predict_proba_two,
                                     num_features=num_features,
                                     num_samples=num_samples, labels=(1,))
    return exp.as_list(label=1), exp


# ----------------------------- helpers --------------------------------------
def top_keywords(shap_df: pd.DataFrame | None, lime_list=None, k: int = 10):
    """Merge SHAP/LIME word scores -> (top REAL-leaning, top FAKE-leaning) DataFrames."""
    scores: dict[str, list[float]] = {}
    if shap_df is not None:
        for tok, val in zip(shap_df["token"], shap_df["value"]):
            w = tok.strip().lower()
            if re.fullmatch(r"[a-z]{3,}", w):
                scores.setdefault(w, []).append(float(val))
    if lime_list:
        for w, val in lime_list:
            scores.setdefault(w.lower(), []).append(float(val))
    df = pd.DataFrame([(w, float(np.sum(v))) for w, v in scores.items()],
                      columns=["word", "score"])
    if df.empty:
        return df, df
    real = df[df.score > 0].nlargest(k, "score").reset_index(drop=True)
    fake = df[df.score < 0].nsmallest(k, "score").reset_index(drop=True)
    return real, fake


def highlight_html(tokens, values) -> str:
    """Inline HTML with words shaded green (-> REAL) or red (-> FAKE)."""
    vmax = max(float(np.max(np.abs(values))), 1e-9)
    parts = []
    for tok, v in zip(tokens, values):
        a = min(abs(float(v)) / vmax, 1.0) * 0.85
        color = f"rgba(46,204,113,{a:.2f})" if v > 0 else f"rgba(231,76,60,{a:.2f})"
        parts.append(f'<span title="{v:+.4f}" style="background:{color};padding:2px 1px;'
                     f'border-radius:3px">{html.escape(tok)}</span>')
    return '<div style="line-height:2.1;font-size:1.02rem">' + "".join(parts) + "</div>"


def importance_figure(words, weights, title: str) -> go.Figure:
    """Horizontal bar chart: green = supports REAL, red = supports FAKE."""
    order = np.argsort(np.abs(weights))
    words, weights = [words[i] for i in order], [weights[i] for i in order]
    fig = go.Figure(go.Bar(x=weights, y=words, orientation="h",
                           marker_color=[GREEN if w > 0 else RED for w in weights]))
    fig.update_layout(title=title, height=max(320, 26 * len(words) + 120),
                      xaxis_title="Contribution to P(REAL)  (← FAKE | REAL →)",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def narrative(label: str, confidence: float, real_df, fake_df) -> str:
    """Plain-English 'why' sentence built from the top keywords."""
    fmt = lambda d: ", ".join(f"'{w}'" for w in d["word"].head(5)) if len(d) else "none"
    lead = f"The model predicts **{label}** with {confidence:.1%} confidence."
    if label == "FAKE":
        return (f"{lead} Words pushing toward FAKE: {fmt(fake_df)}. "
                f"Words pulling toward REAL: {fmt(real_df)}.")
    return (f"{lead} Words pushing toward REAL: {fmt(real_df)}. "
            f"Words pulling toward FAKE: {fmt(fake_df)}.")


def _save_png(words, weights, title, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(np.abs(weights))
    w = [words[i] for i in order]; v = [weights[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(w) + 1)))
    ax.barh(w, v, color=[GREEN if x > 0 else RED for x in v])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(title=title, xlabel="Contribution to P(REAL)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    from src.predict import FakeNewsPredictor

    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    pred = FakeNewsPredictor()
    print(pred.predict(args.text))
    REPORTS_DIR.mkdir(exist_ok=True)

    sdf, _ = explain_shap(pred, args.text)
    real, fake = top_keywords(sdf, None, 10)
    both = pd.concat([real, fake])
    _save_png(both.word.tolist(), both.score.tolist(), "SHAP word importance",
              REPORTS_DIR / "shap_importance.png")
    lime_list, _ = explain_lime(pred, args.text)
    _save_png([w for w, _ in lime_list], [v for _, v in lime_list], "LIME word importance",
              REPORTS_DIR / "lime_importance.png")
    print("Saved reports/shap_importance.png and reports/lime_importance.png")
