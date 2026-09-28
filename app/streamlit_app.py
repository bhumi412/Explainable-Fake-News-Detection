"""
Explainable Fake News Detection — Streamlit dashboard.

Run from the project root:   streamlit run app/streamlit_app.py
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `import src...` works when launched via streamlit

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src import explain as xai
from src.predict import clear_history, load_history, models_ready, save_prediction
from src.preprocess import CLEAN_PATH
from src.report import build_pdf

REPORTS = ROOT / "reports"
COLORS = {"FAKE": "#E74C3C", "REAL": "#2ECC71"}

st.set_page_config(page_title="Explainable Fake News Detection", page_icon="📰", layout="wide")

# ----------------------------- styling (dark-friendly, responsive) -------------
st.markdown("""
<style>
.card{background:rgba(128,128,128,.12);border:1px solid rgba(128,128,128,.28);
      border-radius:14px;padding:16px 20px;margin-bottom:10px}
.card h4{margin:0;font-size:.85rem;opacity:.75;font-weight:500}
.card p{margin:4px 0 0;font-size:1.7rem;font-weight:700}
.badge{display:inline-block;padding:8px 26px;border-radius:999px;font-size:1.6rem;
       font-weight:800;color:#fff}
.badge.FAKE{background:#E74C3C}.badge.REAL{background:#2ECC71}
@media (max-width:768px){
  .block-container{padding:1rem .8rem!important}
  .card p{font-size:1.3rem}.badge{font-size:1.2rem;padding:6px 18px}
}
</style>""", unsafe_allow_html=True)


# ----------------------------- helpers ----------------------------------------
def card(title, value):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{value}</p></div>', unsafe_allow_html=True)


def read_json(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    from src.predict import FakeNewsPredictor
    return FakeNewsPredictor()


@st.cache_data(show_spinner="Loading dataset...")
def load_clean(mtime: float):
    return pd.read_csv(CLEAN_PATH).fillna({"cleaned_text": ""})


@st.cache_data(show_spinner="Counting words...")
def word_counts(mtime: float):
    df = load_clean(mtime)
    out = {}
    for lab, grp in df.groupby("label"):
        c = Counter()
        for t in grp["cleaned_text"]:
            c.update(t.split())
        out[lab] = dict(c.most_common(3000))
    return out


def require_model():
    if not models_ready():
        st.warning("No trained model found. Run `python -m src.train` (see README) or use the "
                   "**Model Training** page, then reload.")
        st.stop()


# ----------------------------- pages ------------------------------------------
def page_home():
    st.title("📰 Explainable Fake News Detection")
    st.caption("Deep Learning (BiLSTM) + Explainable AI (SHAP & LIME)")
    m = read_json("metrics.json")
    if m:
        cols = st.columns(4)
        for c, (k, lab) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Precision"),
                                      ("recall", "Recall"), ("f1", "F1 Score")]):
            with c:
                card(lab, f"{m['metrics'][k]:.2%}")
    left, right = st.columns(2)
    with left:
        st.subheader("Problem statement")
        st.write("Misinformation spreads faster than fact-checkers can respond. Manual "
                 "verification does not scale, and black-box classifiers are hard to trust. "
                 "This project builds a deep-learning classifier that labels articles as "
                 "**FAKE** or **REAL** *and* shows which words drove the decision, so a human "
                 "can audit every prediction.")
        st.subheader("Objectives")
        st.markdown("- Classify news articles with a bidirectional LSTM\n"
                    "- Explain each decision with SHAP and LIME\n"
                    "- Provide analytics, history and downloadable PDF reports")
    with right:
        st.subheader("Pipeline")
        st.markdown("1. **Preprocess** — clean, tokenise, remove stop-words, lemmatise\n"
                    "2. **Vectorise** — Keras Tokenizer + sequence padding\n"
                    "3. **Model** — Embedding → BiLSTM → Dropout → Dense(sigmoid)\n"
                    "4. **Evaluate** — accuracy, precision, recall, F1, ROC, confusion matrix\n"
                    "5. **Explain** — SHAP + LIME word attributions\n"
                    "6. **Serve** — this Streamlit dashboard")
        st.info("Use the sidebar to navigate. Start with **News Detector**.")


def page_dataset():
    st.title("📊 Dataset Insights")
    if not CLEAN_PATH.exists():
        st.warning("Cleaned dataset not found. Put `Fake.csv` and `True.csv` in `data/`, then build it.")
        if st.button("Run preprocessing now"):
            from src.preprocess import build_dataset
            with st.spinner("Cleaning articles (a few minutes)..."):
                try:
                    build_dataset()
                    st.rerun()
                except FileNotFoundError as e:
                    st.error(str(e))
        return
    mtime = CLEAN_PATH.stat().st_mtime
    df = load_clean(mtime)

    c1, c2, c3 = st.columns(3)
    with c1: card("Articles", f"{len(df):,}")
    with c2: card("Fake / Real", f"{(df.label=='FAKE').sum():,} / {(df.label=='REAL').sum():,}")
    with c3: card("Median length (words)", f"{int(df.text_length.median()):,}")

    a, b = st.columns(2)
    with a:
        counts = df.label.value_counts().reset_index()
        counts.columns = ["label", "count"]
        st.plotly_chart(px.pie(counts, names="label", values="count", hole=.45, color="label",
                               color_discrete_map=COLORS, title="Class distribution"),
                        use_container_width=True)
    with b:
        st.plotly_chart(px.histogram(df, x="subject", color="label", barmode="group",
                                     color_discrete_map=COLORS, title="Subjects by class"),
                        use_container_width=True)

    st.plotly_chart(px.histogram(df[df.text_length < df.text_length.quantile(.99)],
                                 x="text_length", color="label", nbins=60, barmode="overlay",
                                 opacity=.65, color_discrete_map=COLORS,
                                 title="Article length (words)"), use_container_width=True)

    wc = word_counts(mtime)
    st.subheader("Most common words")
    n = st.slider("Number of words", 10, 40, 20)
    cols = st.columns(2)
    for col, lab in zip(cols, ["FAKE", "REAL"]):
        top = pd.DataFrame(list(wc[lab].items())[:n], columns=["word", "count"]).iloc[::-1]
        with col:
            st.plotly_chart(px.bar(top, x="count", y="word", orientation="h", title=f"{lab} articles",
                                   color_discrete_sequence=[COLORS[lab]]), use_container_width=True)

    st.subheader("Fake vs Real: distinctive vocabulary")
    tot = {k: sum(v.values()) for k, v in wc.items()}
    rows = []
    for w in set(wc["FAKE"]) & set(wc["REAL"]):
        f, r = wc["FAKE"][w] / tot["FAKE"], wc["REAL"][w] / tot["REAL"]
        if wc["FAKE"][w] + wc["REAL"][w] >= 200:
            rows.append((w, np.log2(f / r)))
    diff = pd.DataFrame(rows, columns=["word", "log2_ratio"]).sort_values("log2_ratio")
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(px.bar(diff.tail(n), x="log2_ratio", y="word", orientation="h",
                               title="Over-represented in FAKE", color_discrete_sequence=[COLORS["FAKE"]]),
                        use_container_width=True)
    with d2:
        st.plotly_chart(px.bar(diff.head(n).iloc[::-1], x="log2_ratio", y="word", orientation="h",
                               title="Over-represented in REAL", color_discrete_sequence=[COLORS["REAL"]]),
                        use_container_width=True)
    st.caption("log2 ratio of relative word frequency (FAKE ÷ REAL); only words with ≥200 total occurrences.")


def page_training():
    st.title("🏋️ Model Training")
    s, h = read_json("training_summary.json"), read_json("history.json")
    if not s:
        st.info("No training run found yet.")
    else:
        cols = st.columns(4)
        with cols[0]: card("Train / Test", f"{s['n_train']:,} / {s['n_test']:,}")
        with cols[1]: card("Epochs run", f"{s['epochs_run']} (best {s['best_epoch']})")
        with cols[2]: card("Best val. accuracy", f"{s['best_val_accuracy']:.2%}")
        with cols[3]: card("Training time", f"{s['training_seconds']/60:.1f} min")
        st.markdown(f"**Parameters:** {s['parameters']:,} · **Batch size:** {s['batch_size']} · "
                    f"**Vocabulary:** {s['max_words']:,} · **Sequence length:** {s['max_len']} · "
                    f"**Validation split:** {s['validation_split']:.0%}")
    m = read_json("metrics.json")
    if m:
        st.subheader("Test-set performance")
        st.dataframe(pd.DataFrame([m["metrics"]]).style.format("{:.4f}"), use_container_width=True)
    if h:
        st.subheader("Per-epoch history")
        st.dataframe(pd.DataFrame(h).round(4), use_container_width=True)

    st.subheader("Architecture")
    st.code("Embedding(20000, 128) → SpatialDropout(0.2) → Bidirectional(LSTM 64) → Dropout(0.5)\n"
            "→ Dense(64, ReLU) → Dropout(0.3) → Dense(1, sigmoid)   [EarlyStopping + ModelCheckpoint]")
    with st.expander("Train from the UI"):
        st.caption("Runs `python -m src.train` in a subprocess (blocks until done; GPU recommended).")
        ep = st.number_input("Epochs", 1, 30, 8)
        sample = st.number_input("Sample size (0 = full dataset)", 0, 100000, 0, step=1000)
        if st.button("Start training"):
            with st.spinner("Training..."):
                p = subprocess.run([sys.executable, "-m", "src.train", "--epochs", str(ep),
                                    "--sample", str(sample)], cwd=ROOT, capture_output=True, text=True)
            st.code((p.stdout[-2500:] + "\n" + p.stderr[-1500:]).strip())
            st.cache_resource.clear()
            st.success("Finished." if p.returncode == 0 else "Training failed — see log above.")


def page_detector():
    st.title("🔎 News Detector")
    require_model()
    text = st.text_area("Paste a news article (headline + body)", height=260,
                        value=st.session_state.get("shared_text", ""),
                        placeholder="WASHINGTON — The Senate voted on Tuesday to ...")
    save = st.checkbox("Save to prediction history", value=True)
    if st.button("Analyze", type="primary"):
        if len(text.split()) < 15:
            st.error("Please enter at least ~15 words for a meaningful prediction.")
        else:
            with st.spinner("Analyzing..."):
                res = get_predictor().predict(text)
            st.session_state.update(shared_text=text, last_result=res, last_text=text)
            if save:
                save_prediction(text, res)

    res = st.session_state.get("last_result")
    if res and st.session_state.get("last_text") == text and text:
        a, b = st.columns([1, 1])
        with a:
            st.markdown(f'<span class="badge {res["label"]}">{res["label"]}</span>', unsafe_allow_html=True)
            st.write("")
            st.progress(res["confidence"], text=f"Confidence: {res['confidence']:.1%}")
            st.caption(f"P(REAL) = {res['prob_real']:.4f}  ·  P(FAKE) = {res['prob_fake']:.4f}")
            st.info("Open **Explainability Dashboard** to see *why* (SHAP & LIME).")
        with b:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=res["confidence"] * 100,
                                         number={"suffix": "%"}, title={"text": "Confidence"},
                                         gauge={"axis": {"range": [0, 100]},
                                                "bar": {"color": COLORS[res["label"]]}}))
            fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        exp = st.session_state.get("explanations", {}).get(_key(text))
        real_df, fake_df, why = (exp["real"], exp["fake"], exp["why"]) if exp else (None, None, "")
        st.download_button("📄 Download PDF report", build_pdf(text, res, real_df, fake_df, why),
                           "fake_news_report.pdf", "application/pdf",
                           help="Tip: run the Explainability Dashboard first to include keywords.")

    st.divider()
    st.subheader("Prediction history")
    hist = load_history()
    if hist.empty:
        st.caption("No predictions saved yet.")
    else:
        st.dataframe(hist, use_container_width=True, hide_index=True)
        c1, c2, _ = st.columns([1, 1, 3])
        c1.download_button("Download CSV", hist.to_csv(index=False), "prediction_history.csv", "text/csv")
        if c2.button("Clear history"):
            clear_history(); st.rerun()


def _key(text):
    return hashlib.md5(text.encode()).hexdigest()


def page_explain():
    st.title("🧠 Explainability Dashboard")
    require_model()
    st.caption("Green = pushes toward REAL · Red = pushes toward FAKE (values are contributions to P(REAL)).")
    text = st.text_area("Article to explain", height=200, value=st.session_state.get("shared_text", ""))
    c1, c2, c3 = st.columns(3)
    max_words = c1.slider("SHAP: words analysed", 40, 200, 100, 10)
    n_samples = c2.select_slider("LIME: perturbation samples", [200, 500, 1000, 2000], 500)
    n_feat = c3.slider("Top features", 5, 25, 12)

    if st.button("Generate explanations", type="primary"):
        if len(text.split()) < 15:
            st.error("Please enter at least ~15 words.")
        else:
            st.session_state["shared_text"] = text
            pred = get_predictor()
            with st.spinner("Computing SHAP (can take up to a minute on CPU)..."):
                sdf, base = xai.explain_shap(pred, text, max_words=max_words)
            with st.spinner("Computing LIME..."):
                lime_list, lime_exp = xai.explain_lime(pred, text, n_feat, n_samples)
            res = pred.predict(text)
            real, fake = xai.top_keywords(sdf, lime_list, n_feat)
            st.session_state.setdefault("explanations", {})[_key(text)] = dict(
                res=res, sdf=sdf, base=base, lime=lime_list, lime_html=lime_exp.as_html(labels=(1,)),
                real=real, fake=fake, why=xai.narrative(res["label"], res["confidence"], real, fake))
            st.session_state["last_result"], st.session_state["last_text"] = res, text

    exp = st.session_state.get("explanations", {}).get(_key(text)) if text else None
    if not exp:
        st.info("Paste an article and click **Generate explanations**.")
        return
    res = exp["res"]
    st.markdown(f'<span class="badge {res["label"]}">{res["label"]}</span> &nbsp; '
                f'confidence **{res["confidence"]:.1%}**', unsafe_allow_html=True)
    st.success(exp["why"])

    t1, t2, t3 = st.tabs(["SHAP", "LIME", "Important keywords"])
    with t1:
        sdf = exp["sdf"]
        st.markdown("**Word highlights**")
        st.markdown(xai.highlight_html(sdf["token"], sdf["value"]), unsafe_allow_html=True)
        st.caption(f"Base value (all words masked): P(REAL) = {exp['base']:.3f}. Hover a word for its SHAP value.")
        g = sdf.assign(word=sdf.token.str.strip().str.lower())
        g = g[g.word.str.fullmatch(r"[a-z]{3,}")].groupby("word", as_index=False).value.sum()
        g = g.reindex(g.value.abs().sort_values(ascending=False).index).head(n_feat)
        st.plotly_chart(xai.importance_figure(g.word.tolist(), g.value.tolist(), "SHAP feature importance"),
                        use_container_width=True)
    with t2:
        words, weights = zip(*exp["lime"]) if exp["lime"] else ([], [])
        st.plotly_chart(xai.importance_figure(list(words), list(weights), "LIME feature importance"),
                        use_container_width=True)
        with st.expander("Original LIME widget"):
            components.html(exp["lime_html"], height=420, scrolling=True)
    with t3:
        a, b = st.columns(2)
        a.markdown("#### 🟢 Supports REAL"); a.dataframe(exp["real"], hide_index=True, use_container_width=True)
        b.markdown("#### 🔴 Supports FAKE"); b.dataframe(exp["fake"], hide_index=True, use_container_width=True)
        st.caption("Scores combine SHAP and LIME contributions for the same word.")

    st.download_button("📄 Download PDF report (with explanation)",
                       build_pdf(text, res, exp["real"], exp["fake"], exp["why"]),
                       "fake_news_explained.pdf", "application/pdf")


def page_analytics():
    st.title("📈 Model Analytics")
    m, h = read_json("metrics.json"), read_json("history.json")
    if not m:
        st.info("Train the model first to see analytics.")
        return
    cols = st.columns(5)
    for c, (k, lab) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Precision"),
                                  ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC AUC")]):
        with c: card(lab, f"{m['metrics'][k]:.4f}")
    if h:
        ep = list(range(1, len(h["loss"]) + 1))
        a, b = st.columns(2)
        for col, key, title in ((a, "accuracy", "Accuracy"), (b, "loss", "Loss")):
            f = go.Figure()
            f.add_scatter(x=ep, y=h[key], mode="lines+markers", name="train")
            f.add_scatter(x=ep, y=h["val_" + key], mode="lines+markers", name="validation")
            f.update_layout(title=title, xaxis_title="Epoch", height=360)
            col.plotly_chart(f, use_container_width=True)
    a, b = st.columns(2)
    roc = m["roc"]
    f = go.Figure()
    f.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"AUC = {roc['auc']:.4f}", line_width=3)
    f.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="chance")
    f.update_layout(title="ROC curve", xaxis_title="False positive rate",
                    yaxis_title="True positive rate", height=420)
    a.plotly_chart(f, use_container_width=True)
    cm = np.array(m["confusion_matrix"])
    f = px.imshow(cm, text_auto=True, x=m["classes"], y=m["classes"], color_continuous_scale="Blues",
                  labels=dict(x="Predicted", y="Actual"), title="Confusion matrix")
    f.update_layout(height=420)
    b.plotly_chart(f, use_container_width=True)
    with st.expander("Per-class report"):
        st.dataframe(pd.DataFrame(m["classification_report"]).T.round(4), use_container_width=True)


# ----------------------------- router -----------------------------------------
PAGES = {"🏠 Home": page_home, "📊 Dataset Insights": page_dataset, "🏋️ Model Training": page_training,
         "🔎 News Detector": page_detector, "🧠 Explainability Dashboard": page_explain,
         "📈 Model Analytics": page_analytics}
with st.sidebar:
    st.header("Navigation")
    choice = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Model: BiLSTM · XAI: SHAP + LIME\n\nTheme: use ⋮ → Settings to switch light/dark.")
PAGES[choice]()
