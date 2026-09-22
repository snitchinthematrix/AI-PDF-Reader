"""Pure text-processing helpers.

No PDF or ML dependencies here on purpose, so these functions can be
unit-tested with nothing but the standard library.
"""
from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")
_NON_SENTENCE_PUNCTUATION_RE = re.compile(r"[^\w\s.,!?-]")


def preprocess_text(text: str) -> str:
    """Collapse whitespace and drop characters outside basic sentence punctuation."""
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NON_SENTENCE_PUNCTUATION_RE.sub("", text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 500) -> list[str]:
    """Group sentences into chunks of roughly `chunk_size` characters each.

    Splitting on sentence boundaries (rather than a hard character cutoff)
    keeps each chunk readable and avoids cutting a sentence in half.
    """
    if not text:
        return []

    sentences = [s.strip() for s in text.split(".") if s.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_size = 0

    for sentence in sentences:
        current_chunk.append(sentence + ".")
        current_size += len(sentence)

        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
