"""Local, offline NLP analysis: sentiment, key-point extraction, and question answering.

The heavy ML libraries (torch, transformers, sentence-transformers) are imported
lazily inside `DocumentAnalyzer.__init__`, and the three underlying models can be
injected directly. That keeps this module importable - and its logic unit-testable
with lightweight fakes - without those multi-gigabyte dependencies installed.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np

logger = logging.getLogger(__name__)

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QA_MODEL = "distilbert-base-cased-distilled-squad"


class RhetoricAnalysis(TypedDict):
    sentiment: dict
    key_points: list[str]


def _cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single vector and each row of a matrix."""
    vector_norm = vector / (np.linalg.norm(vector) + 1e-10)
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix_norm = matrix / matrix_norms
    return matrix_norm @ vector_norm


class DocumentAnalyzer:
    """Wraps the local models used to analyze extracted document text.

    Real Hugging Face / sentence-transformers models are loaded by default.
    Pass `sentiment_analyzer`, `embedding_model`, and/or `qa_pipeline` to inject
    fakes (this is how the unit tests avoid downloading real models).
    """

    def __init__(
        self,
        sentiment_analyzer: Callable | None = None,
        embedding_model: Any | None = None,
        qa_pipeline: Callable | None = None,
    ) -> None:
        if sentiment_analyzer is None or embedding_model is None or qa_pipeline is None:
            logger.info("Loading local NLP models (this can take a moment on first run)...")
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline

            sentiment_analyzer = sentiment_analyzer or pipeline(
                "sentiment-analysis", model=SENTIMENT_MODEL, device=-1
            )
            embedding_model = embedding_model or SentenceTransformer(EMBEDDING_MODEL)
            qa_pipeline = qa_pipeline or pipeline(
                "question-answering", model=QA_MODEL, device=-1
            )

        self.sentiment_analyzer = sentiment_analyzer
        self.embedding_model = embedding_model
        self.qa_pipeline = qa_pipeline

    def analyze_rhetoric(self, text_chunk: str, top_k: int = 3) -> RhetoricAnalysis:
        """Return sentiment plus the `top_k` sentences most representative of `text_chunk`."""
        sentiment = self.sentiment_analyzer(text_chunk)[0]

        sentences = [s.strip() for s in text_chunk.split(".") if s.strip()]
        if not sentences:
            return {"sentiment": sentiment, "key_points": []}

        sentence_embeddings = np.asarray(self.embedding_model.encode(sentences))
        text_embedding = np.asarray(self.embedding_model.encode(text_chunk))

        similarities = _cosine_similarity(text_embedding, sentence_embeddings)
        top_n = min(top_k, len(sentences))
        top_indices = np.argsort(similarities)[::-1][:top_n]
        key_points = [sentences[i] for i in top_indices]

        return {"sentiment": sentiment, "key_points": key_points}

    def answer_question(self, question: str, context: str) -> str:
        """Answer a question about `context`, or return a friendly fallback on failure."""
        if not question.strip() or not context.strip():
            return "Please provide both a question and some text to search."
        try:
            result = self.qa_pipeline(question=question, context=context)
            return result["answer"]
        except Exception:
            logger.exception("QA pipeline failed")
            return "I couldn't find a specific answer to that question. Try rephrasing it?"
