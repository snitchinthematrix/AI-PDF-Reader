"""Streamlit UI for the local PDF Reader & Analysis tool."""
from __future__ import annotations

import io
import logging

import streamlit as st

from pdf_reader.analyzer import DocumentAnalyzer
from pdf_reader.pdf_extraction import PDFExtractionError, extract_text_from_pdf
from pdf_reader.text_utils import preprocess_text, split_into_chunks

logging.basicConfig(level=logging.INFO)


@st.cache_resource(show_spinner="Loading local NLP models (first run only)...")
def get_analyzer() -> DocumentAnalyzer:
    """Load the ML models once per session instead of on every rerun/click."""
    return DocumentAnalyzer()


@st.cache_data(show_spinner="Extracting text from PDF...")
def process_pdf(file_bytes: bytes) -> list[str]:
    """Extract, clean, and chunk PDF text. Cached by file content."""
    text = extract_text_from_pdf(io.BytesIO(file_bytes))
    processed = preprocess_text(text)
    return split_into_chunks(processed)


def main() -> None:
    st.set_page_config(page_title="PDF Reader & Analysis", page_icon="📄")
    st.title("📄 PDF Reader & Analysis")
    st.write("Upload a PDF to analyze its content — 100% local, no API keys needed.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is None:
        return

    try:
        with st.spinner("Processing PDF..."):
            chunks = process_pdf(uploaded_file.getvalue())
    except PDFExtractionError as exc:
        st.error(str(exc))
        return

    if not chunks:
        st.warning("No extractable text found in this PDF (it may be a scanned image).")
        return

    st.subheader("Document Text")
    chunk_selector = st.selectbox(
        "Select text chunk to analyze:",
        range(len(chunks)),
        format_func=lambda x: f"Chunk {x + 1} of {len(chunks)}",
    )
    st.text_area("Text Content", chunks[chunk_selector], height=200)

    analyzer = get_analyzer()

    if st.button("Analyze This Chunk"):
        with st.spinner("Analyzing..."):
            analysis = analyzer.analyze_rhetoric(chunks[chunk_selector])

        st.subheader("Analysis Results")
        st.write("Sentiment:", analysis["sentiment"]["label"])
        st.write("Confidence:", f"{analysis['sentiment']['score']:.2%}")

        st.write("Key Points:")
        for idx, point in enumerate(analysis["key_points"], 1):
            st.write(f"{idx}. {point}")

    st.subheader("Ask Questions")
    question = st.text_input("Ask a question about the text:")
    if question and st.button("Get Answer"):
        with st.spinner("Finding answer..."):
            answer = analyzer.answer_question(question, chunks[chunk_selector])
        st.write("Answer:", answer)


if __name__ == "__main__":
    main()
