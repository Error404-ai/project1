"""
model_loader.py
Loads HuggingFace pipelines on demand and releases them immediately after use,
so peak memory is bounded to ONE model at a time instead of all models
accumulating in memory for the life of the process.

Why this changed from a permanent lru_cache:
- On a 512Mi container, holding summarizer + sentiment + QA models all in
  memory simultaneously (as happens after a few /api/analyze calls with a
  permanent cache) exceeds available RAM and gets OOM-killed by the host.
- Trade-off: every call now reloads weights from disk, which is slower
  (extra seconds per request) but survives on constrained memory. If you
  upgrade to a larger instance, flip LOW_MEMORY_MODE to False to restore
  the old "load once, keep forever" behavior for speed.

Memory optimization (unchanged):
- CPU-only torch avoids pulling in unused CUDA libs.
- Each model is dynamically quantized to INT8 after loading, cutting RAM
  usage roughly 3-4x vs full fp32 weights.
- The fp32 copy is explicitly deleted and garbage-collected right after
  quantization so it doesn't linger during the highest-memory moment.

Models used (all free, downloaded automatically on first run):
- Summarization : sshleifer/distilbart-cnn-6-6   (smaller than 12-6, survives 512Mi much better)
- Sentiment     : distilbert-base-uncased-finetuned-sst-2-english
- Question-Ans  : distilbert-base-cased-distilled-squad
"""

import gc
import os

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    pipeline,
)

# Keep CPU thread usage predictable on small/shared instances
torch.set_num_threads(1)

# Set LOW_MEMORY_MODE=false as an env var once you're on a bigger instance
# to go back to "load once, cache forever" for speed.
LOW_MEMORY_MODE = os.environ.get("LOW_MEMORY_MODE", "true").lower() != "false"

# Switched from distilbart-cnn-12-6 (~306M params, ~1.2GB fp32) to the
# 6-6 variant (~230M params, ~920MB fp32) - still the single biggest
# consumer here, see note in README/comments below if you're still OOMing.
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-6-6"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
QA_MODEL = "distilbert-base-cased-distilled-squad"

_cache = {}


def _quantize(model):
    """Dynamic INT8 quantization of Linear layers - lossless in practice for
    these encoder/seq2seq architectures, big win for CPU inference memory."""
    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    del model
    gc.collect()
    return quantized


def _load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARIZATION_MODEL, low_cpu_mem_usage=True
    )
    model = _quantize(model)
    return pipeline("summarization", model=model, tokenizer=tokenizer)


def _load_sentiment_analyzer():
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL, low_cpu_mem_usage=True
    )
    model = _quantize(model)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def _load_qa_model():
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(
        QA_MODEL, low_cpu_mem_usage=True
    )
    model = _quantize(model)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)


def _get(key, loader_fn):
    """
    In LOW_MEMORY_MODE: load fresh every time, never cache. Caller is
    responsible for calling release_all() after use.
    Otherwise: classic cache-forever behavior (fast, needs more RAM).
    """
    if LOW_MEMORY_MODE:
        return loader_fn()
    if key not in _cache:
        _cache[key] = loader_fn()
    return _cache[key]


def release_all():
    """
    Drop any cached models and force garbage collection. Call this after
    each pipeline step in LOW_MEMORY_MODE so the next model has the full
    memory budget available rather than stacking on top of the last one.
    In non-low-memory mode this is a no-op by default (models stay cached);
    call it manually only if you actually want to free RAM.
    """
    if LOW_MEMORY_MODE:
        _cache.clear()
        gc.collect()


def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """
    Summarizes text using a transformer-based abstractive summarizer.
    Long articles are chunked because the model has a token limit (~1024 tokens).
    """
    summarizer = _get("summarizer", _load_summarizer)
    try:
        max_chunk_chars = 3000
        if len(text) <= max_chunk_chars:
            result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return result[0]["summary_text"]

        chunks = [text[i:i + max_chunk_chars] for i in range(0, len(text), max_chunk_chars)]
        partial_summaries = []
        for chunk in chunks:
            if len(chunk.split()) < 10:
                continue
            out = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            partial_summaries.append(out[0]["summary_text"])

        combined = " ".join(partial_summaries)
        if len(combined) > max_chunk_chars:
            final = summarizer(combined, max_length=max_length, min_length=min_length, do_sample=False)
            return final[0]["summary_text"]
        return combined
    finally:
        release_all()


def analyze_sentiment(text: str) -> dict:
    """Returns sentiment label (POSITIVE/NEGATIVE) and confidence score."""
    analyzer = _get("sentiment", _load_sentiment_analyzer)
    try:
        truncated = text[:2000]
        result = analyzer(truncated)[0]
        return {"label": result["label"], "confidence": round(float(result["score"]), 4)}
    finally:
        release_all()


def answer_question(context: str, question: str) -> dict:
    """
    Answers a user question using an extractive QA model.
    """
    qa_model = _get("qa", _load_qa_model)
    try:
        prompt_context = context[:4000]
        result = qa_model(question=question, context=prompt_context)
        return {
            "answer": result["answer"],
            "confidence": round(float(result["score"]), 4),
        }
    finally:
        release_all()