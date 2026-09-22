"""Unit tests for DocumentAnalyzer.

These inject lightweight fakes for the sentiment/embedding/QA models, so the
tests run with just numpy + pytest installed - no torch/transformers/
sentence-transformers download required.
"""
import numpy as np
import pytest

from pdf_reader.analyzer import DocumentAnalyzer


class FakeEmbeddingModel:
    """Returns pre-set vectors for known strings so similarity is deterministic."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, text):
        if isinstance(text, list):
            return np.array([self.vectors[t] for t in text])
        return np.array(self.vectors[text])


def _analyzer(sentiment=None, embedding_model=None, qa=None) -> DocumentAnalyzer:
    return DocumentAnalyzer(
        sentiment_analyzer=sentiment or (lambda text: [{"label": "POSITIVE", "score": 0.99}]),
        embedding_model=embedding_model or FakeEmbeddingModel({}),
        qa_pipeline=qa or (lambda **kwargs: {"answer": "unused"}),
    )


def test_analyze_rhetoric_returns_sentiment_and_most_representative_sentence():
    text_chunk = "Alpha content here. Beta content here. Gamma content here."
    vectors = {
        text_chunk: [1.0, 0.0, 0.0],
        "Alpha content here": [1.0, 0.0, 0.0],
        "Beta content here": [0.0, 1.0, 0.0],
        "Gamma content here": [0.0, 0.0, 1.0],
    }
    analyzer = _analyzer(embedding_model=FakeEmbeddingModel(vectors))

    result = analyzer.analyze_rhetoric(text_chunk, top_k=1)

    assert result["sentiment"] == {"label": "POSITIVE", "score": 0.99}
    assert result["key_points"] == ["Alpha content here"]


def test_analyze_rhetoric_handles_text_with_no_sentences():
    analyzer = _analyzer()

    result = analyzer.analyze_rhetoric("...")

    assert result["key_points"] == []


def test_answer_question_returns_model_answer():
    analyzer = _analyzer(qa=lambda question, context: {"answer": "42"})

    assert analyzer.answer_question("What is the answer?", "Some context.") == "42"


def test_answer_question_requires_question_and_context():
    analyzer = _analyzer()

    assert "provide both" in analyzer.answer_question("", "some context").lower()
    assert "provide both" in analyzer.answer_question("a question", "  ").lower()


def test_answer_question_falls_back_on_pipeline_error():
    def broken_pipeline(**kwargs):
        raise RuntimeError("boom")

    analyzer = _analyzer(qa=broken_pipeline)

    answer = analyzer.answer_question("Q?", "context")

    assert "couldn't find" in answer.lower()
