"""
nlp_utils.py
Handles classic NLP preprocessing, keyword extraction, and named-entity extraction.
These are separate from the deep-learning pipelines in model_loader.py so that
preprocessing can run fast and independently of the heavier transformer models.
"""

import re
import string
from functools import lru_cache

import nltk
import spacy
import yake

# ---------------------------------------------------------------------------
# One-time downloads
# ---------------------------------------------------------------------------
# NLTK data needed for tokenization + stopwords
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

STOPWORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# spaCy is used for Named Entity Recognition (people, orgs, locations, dates, etc.)
# Loaded lazily on first use (not at import time) so app startup is fast and
# memory stays low until an /api/entities or /api/analyze request actually
# needs it. Run `python -m spacy download en_core_web_sm` once before starting
# the server, or install it via requirements.txt as a direct wheel URL.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        )


def clean_text(raw_text: str) -> str:
    """Basic normalization: strip extra whitespace/newlines, remove junk characters."""
    text = raw_text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)  # strip URLs
    return text


def preprocess(raw_text: str) -> dict:
    """
    Full preprocessing pipeline:
    - cleaning
    - sentence tokenization
    - word tokenization
    - stopword + punctuation removal
    Returns useful stats and the cleaned text used downstream by other models.
    """
    cleaned = clean_text(raw_text)
    sentences = sent_tokenize(cleaned)
    words = word_tokenize(cleaned)

    filtered_words = [
        w.lower() for w in words
        if w.lower() not in STOPWORDS and w not in string.punctuation
    ]

    return {
        "cleaned_text": cleaned,
        "num_sentences": len(sentences),
        "num_words": len(words),
        "num_significant_words": len(filtered_words),
        "sentences": sentences,
    }


def extract_keywords(text: str, max_keywords: int = 10) -> list:
    """
    Extracts top keywords/keyphrases using YAKE (Yet Another Keyword Extractor).
    YAKE is unsupervised and doesn't require a pretrained model download,
    making it fast and reliable for this pipeline.
    """
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=2,               # up to 2-word phrases
        dedupLim=0.7,
        top=max_keywords,
        features=None,
    )
    keywords = kw_extractor.extract_keywords(text)
    # yake returns (keyword, score) where LOWER score = more relevant
    keywords.sort(key=lambda x: x[1])
    return [{"keyword": kw, "score": round(float(score), 4)} for kw, score in keywords]


def extract_entities(text: str) -> list:
    """Extracts named entities (people, organizations, locations, dates, etc.) using spaCy."""
    nlp = get_nlp()
    doc = nlp(text[:100000])  # guard against extremely long inputs
    entities = [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]
    # de-duplicate while preserving order
    seen = set()
    unique_entities = []
    for e in entities:
        key = (e["text"].lower(), e["label"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(e)
    return unique_entities