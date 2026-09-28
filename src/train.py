"""
Model training: Tokenizer -> padding -> Embedding -> BiLSTM -> Dropout -> Dense(sigmoid).

Run:  python -m src.train --epochs 8 --batch-size 128
Quick smoke test:  python -m src.train --sample 4000 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocess import load_cleaned, pad

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR, REPORTS_DIR = ROOT / "models", ROOT / "reports"
MODEL_PATH = MODELS_DIR / "fake_news_model.h5"
SEED = 42


def build_model(max_words: int, max_len: int, embed_dim: int = 128, lstm_units: int = 64):
    """Embedding -> SpatialDropout -> BiLSTM -> Dropout -> Dense -> Dropout -> sigmoid."""
    from tensorflow.keras import Input, Sequential
    from tensorflow.keras.layers import (Bidirectional, Dense, Dropout, Embedding,
                                         LSTM, SpatialDropout1D)
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(max_words, embed_dim),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(lstm_units)),          # reads the article in both directions
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),           # P(REAL)
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main(args):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.preprocessing.text import Tokenizer

    tf.keras.utils.set_random_seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_cleaned()
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=SEED)
    print(f"Dataset: {len(df):,} articles\n{df['label'].value_counts().to_string()}")

    # FAKE -> 0, REAL -> 1 (alphabetical); model outputs P(REAL)
    le = LabelEncoder().fit(["FAKE", "REAL"])
    y = le.transform(df["label"])

    # ----- train / test split (stratified) ----------------------------------
    X = df["cleaned_text"].astype(str).to_numpy()
    y = np.array(y)

    X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )

    # ----- vectorisation: fit tokenizer on TRAIN only (no leakage) ----------
    tokenizer = Tokenizer(num_words=args.max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_tr_txt)
    X_tr = pad(tokenizer.texts_to_sequences(X_tr_txt), args.max_len)
    X_te = pad(tokenizer.texts_to_sequences(X_te_txt), args.max_len)

    model = build_model(args.max_words, args.max_len)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True),
    ]
    t0 = time.time()
    history = model.fit(X_tr, y_tr, validation_split=0.1, epochs=args.epochs,
                        batch_size=args.batch_size, callbacks=callbacks, verbose=1)
    seconds = time.time() - t0

    # ----- persist artefacts ------------------------------------------------
    joblib.dump(tokenizer, MODELS_DIR / "tokenizer.pkl")
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
    np.savez_compressed(MODELS_DIR / "test_data.npz", X_test=X_te, y_test=y_te)
    (MODELS_DIR / "config.json").write_text(json.dumps(
        {"max_words": args.max_words, "max_len": args.max_len, "threshold": 0.5}, indent=2))

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (REPORTS_DIR / "history.json").write_text(json.dumps(hist, indent=2))
    best = int(np.argmin(hist["val_loss"]))
    (REPORTS_DIR / "training_summary.json").write_text(json.dumps({
        "n_articles": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "validation_split": 0.1, "epochs_planned": args.epochs,
        "epochs_run": len(hist["loss"]), "best_epoch": best + 1,
        "best_val_loss": hist["val_loss"][best], "best_val_accuracy": hist["val_accuracy"][best],
        "training_seconds": round(seconds, 1), "batch_size": args.batch_size,
        "max_words": args.max_words, "max_len": args.max_len,
        "parameters": int(model.count_params()),
    }, indent=2))

    from src.evaluate import evaluate
    evaluate()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-words", type=int, default=20000, help="vocabulary size")
    p.add_argument("--max-len", type=int, default=300, help="tokens per article")
    p.add_argument("--sample", type=int, default=0, help="use a subset (quick test)")
    main(p.parse_args())
