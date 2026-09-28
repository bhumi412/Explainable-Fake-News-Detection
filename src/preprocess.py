"""
Data loading and text preprocessing.

Pipeline: missing values -> lowercase -> strip URLs/special chars -> tokenise
-> stop-word removal -> lemmatisation -> `cleaned_text` column.

Run:  python -m src.preprocess
"""
from __future__ import annotations

import re
import time
from functools import lru_cache
from pathlib import Path

import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CLEAN_PATH = DATA_DIR / "cleaned_news.csv"

# Regex tokenizer avoids the need for the NLTK "punkt" download.
_TOKENIZER = RegexpTokenizer(r"[a-z]+")

# Publisher artefacts that leak the label in this dataset (e.g. every REAL article
# starts with "CITY (Reuters) -"). Removing them forces the model to learn content.
_LEAKY_WORDS = {"reuters", "via", "getty", "images", "image", "featured",
                "photo", "pic", "twitter", "com", "https", "www", "http"}


@lru_cache(maxsize=1)
def _resources():
    """Load (and download if needed) NLTK resources once per process."""
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    probes = (("stopwords", lambda: stopwords.words("english")),
              ("wordnet", lambda: WordNetLemmatizer().lemmatize("cars")))
    for pkg, probe in probes:
        try:
            probe()
        except LookupError:
            nltk.download(pkg, quiet=True)
    stops = set(stopwords.words("english")) | _LEAKY_WORDS
    return stops, WordNetLemmatizer()


@lru_cache(maxsize=300_000)
def _lemma(token: str) -> str:
    return _resources()[1].lemmatize(token)  # cached: huge speed-up on 40k+ articles


def clean_text(text) -> str:
    """Return the cleaned, lemmatised version of `text` ('' for missing values)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    stops = _resources()[0]
    text = text.lower()
    text = re.sub(r"^.{0,80}?\(reuters\)\s*-?\s*", " ", text)   # dateline leakage
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)            # URLs
    text = re.sub(r"[^a-z\s]", " ", text)                         # special chars/digits
    tokens = _TOKENIZER.tokenize(text)
    return " ".join(_lemma(t) for t in tokens if len(t) > 2 and t not in stops)


def build_dataset(data_dir: Path = DATA_DIR, save: bool = True) -> pd.DataFrame:
    """Load Fake.csv / True.csv, clean everything and (optionally) cache to CSV."""
    fake_p, true_p = Path(data_dir) / "Fake.csv", Path(data_dir) / "True.csv"
    if not (fake_p.exists() and true_p.exists()):
        raise FileNotFoundError(
            f"Place Fake.csv and True.csv inside {Path(data_dir).resolve()} "
            "(Kaggle: 'Fake and Real News Dataset')."
        )
    t0 = time.time()
    fake, true = pd.read_csv(fake_p), pd.read_csv(true_p)
    fake["label"], true["label"] = "FAKE", "REAL"
    df = pd.concat([fake, true], ignore_index=True)

    # --- missing values ---------------------------------------------------
    for col in ("title", "text", "subject", "date"):
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df = df.drop_duplicates(subset=["title", "text"])
    df = df[(df["title"].str.strip() != "") | (df["text"].str.strip() != "")]

    # --- cleaning ---------------------------------------------------------
    content = df["title"] + " " + df["text"]
    df["text_length"] = content.str.split().str.len()
    print(f"Cleaning {len(df):,} articles (first run takes a few minutes)...")
    df["cleaned_text"] = content.map(clean_text)
    df = df[df["cleaned_text"].str.len() > 0]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df[["title", "subject", "date", "label", "text_length", "cleaned_text"]]

    if save:
        df.to_csv(CLEAN_PATH, index=False)
        print(f"Saved {CLEAN_PATH} ({len(df):,} rows) in {time.time() - t0:.0f}s")
    return df


def load_cleaned(rebuild: bool = False) -> pd.DataFrame:
    """Load the cached cleaned dataset, building it first if necessary."""
    if rebuild or not CLEAN_PATH.exists():
        return build_dataset()
    return pd.read_csv(CLEAN_PATH).fillna({"cleaned_text": ""})


def pad(sequences, maxlen: int):
    """Pad/truncate integer sequences. 'pre' padding keeps real tokens next to the
    LSTM output; 'post' truncation keeps the beginning of the article."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(sequences, maxlen=maxlen, padding="pre", truncating="post")


def texts_to_padded(tokenizer, texts, maxlen: int):
    """Raw texts -> clean -> integer sequences -> padded matrix (used at inference)."""
    return pad(tokenizer.texts_to_sequences([clean_text(t) for t in texts]), maxlen)


if __name__ == "__main__":
    build_dataset()
