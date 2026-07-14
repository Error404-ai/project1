"""
main.py
FastAPI application for the Intelligent News Summarization and Sentiment Analysis System.

Endpoints:
  POST /api/preprocess  - clean text + basic stats
  POST /api/summarize   - abstractive summary
  POST /api/sentiment   - sentiment label + confidence
  POST /api/keywords    - top keywords/keyphrases
  POST /api/entities    - named entities
  POST /api/qa          - question answering over the article
  POST /api/analyze     - runs the full pipeline in one call (used by the frontend)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

import nlp_utils
import model_loader

app = FastAPI(
    title="Intelligent News Summarization & Sentiment Analysis API",
    description="NLP + LLM based system for summarizing news, detecting sentiment, "
                "extracting entities/keywords, and answering questions about an article.",
    version="1.0.0",
)

# Allow the frontend (served from a different origin/port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ArticleRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Full news article text")


class QARequest(BaseModel):
    text: str = Field(..., min_length=20)
    question: str = Field(..., min_length=3)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=20)
    question: Optional[str] = None


@app.get("/")
def health_check():
    return {"status": "ok", "message": "News NLP API is running"}


@app.post("/api/preprocess")
def preprocess_endpoint(req: ArticleRequest):
    try:
        return nlp_utils.preprocess(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize")
def summarize_endpoint(req: ArticleRequest):
    try:
        summary = model_loader.summarize_text(req.text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sentiment")
def sentiment_endpoint(req: ArticleRequest):
    try:
        return model_loader.analyze_sentiment(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/keywords")
def keywords_endpoint(req: ArticleRequest):
    try:
        return {"keywords": nlp_utils.extract_keywords(req.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/entities")
def entities_endpoint(req: ArticleRequest):
    try:
        return {"entities": nlp_utils.extract_entities(req.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qa")
def qa_endpoint(req: QARequest):
    try:
        return model_loader.answer_question(req.text, req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    """
    Runs the complete pipeline in a single call:
    preprocessing -> summarization -> sentiment -> keywords -> entities -> (optional) QA
    """
    try:
        preprocessed = nlp_utils.preprocess(req.text)
        cleaned = preprocessed["cleaned_text"]

        summary = model_loader.summarize_text(cleaned)
        sentiment = model_loader.analyze_sentiment(cleaned)
        keywords = nlp_utils.extract_keywords(cleaned)
        entities = nlp_utils.extract_entities(cleaned)

        response = {
            "stats": {
                "num_sentences": preprocessed["num_sentences"],
                "num_words": preprocessed["num_words"],
            },
            "summary": summary,
            "sentiment": sentiment,
            "keywords": keywords,
            "entities": entities,
        }

        if req.question:
            response["qa"] = model_loader.answer_question(cleaned, req.question)

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)