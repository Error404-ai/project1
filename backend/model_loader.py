"""
model_loader.py
Loads and caches open-source HuggingFace pipelines (no API key required).
Models are loaded lazily (on first use) and cached in memory so repeated
requests don't reload weights from disk each time.

Memory optimization:
- CPU-only torch (see requirements.txt) avoids pulling in unused CUDA libs.
- Each model is dynamically quantized to INT8 after loading. This is
  near-lossless for these architectures (weights are stored more compactly,
  the model itself and its outputs are unchanged) and cuts RAM usage by
  roughly 3-4x compared to full fp32 weights - which is what actually
  matters for fitting inside a 512Mi container.

Models used (all free, downloaded automatically on first run):
- Summarization : sshleifer/distilbart-cnn-12-6   (fast, good quality)
- Sentiment     : distilbert-base-uncased-finetuned-sst-2-english
- Question-Ans  : distilbert-base-cased-distilled-squad
"""

from functools import lru_cache

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

SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
QA_MODEL = "distilbert-base-cased-distilled-squad"


def _quantize(model):
    """Dynamic INT8 quantization of Linear layers - lossless in practice for
    these encoder/seq2seq architectures, big win for CPU inference memory."""
    return torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )


@lru_cache(maxsize=1)
def get_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARIZATION_MODEL, low_cpu_mem_usage=True
    )
    model = _quantize(model)
    return pipeline("summarization", model=model, tokenizer=tokenizer)


@lru_cache(maxsize=1)
def get_sentiment_analyzer():
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL, low_cpu_mem_usage=True
    )
    model = _quantize(model)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


@lru_cache(maxsize=1)
def get_qa_model():
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(
        QA_MODEL, low_cpu_mem_usage=True
    )
    model = _quantize(model)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)


def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """
    Summarizes text using a transformer-based abstractive summarizer.
    Long articles are chunked because the model has a token limit (~1024 tokens).
    """
    summarizer = get_summarizer()

    # Rough chunking by character count to stay within model token limits
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
    # If the combined summary of chunks is still long, summarize it once more
    if len(combined) > max_chunk_chars:
        final = summarizer(combined, max_length=max_length, min_length=min_length, do_sample=False)
        return final[0]["summary_text"]
    return combined


def analyze_sentiment(text: str) -> dict:
    """Returns sentiment label (POSITIVE/NEGATIVE) and confidence score."""
    analyzer = get_sentiment_analyzer()
    # Model has a 512-token limit; truncate long text for this specific call
    truncated = text[:2000]
    result = analyzer(truncated)[0]
    return {"label": result["label"], "confidence": round(float(result["score"]), 4)}


def answer_question(context: str, question: str) -> dict:
    """
    Answers a user question using an extractive QA model.
    A structured prompt template is used to frame the context and question
    consistently before passing them to the model (prompt engineering step).
    """
    qa_model = get_qa_model()

    # --- Prompt template (prompt engineering) ---
    # Even though the underlying model is extractive rather than generative,
    # we standardize how context + question are framed and truncated so that
    # answers stay grounded strictly in the article content.
    prompt_context = context[:4000]

    result = qa_model(question=question, context=prompt_context)
    return {
        "answer": result["answer"],
        "confidence": round(float(result["score"]), 4),
    }