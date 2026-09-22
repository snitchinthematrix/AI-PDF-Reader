"""PDF text extraction via PyMuPDF."""
from __future__ import annotations

from typing import BinaryIO

import fitz  # PyMuPDF


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


def extract_text_from_pdf(pdf_file: BinaryIO) -> str:
    """Extract and concatenate the text of every page in `pdf_file`."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    except Exception as exc:  # PyMuPDF raises plain Exception/RuntimeError on bad input
        raise PDFExtractionError(f"Could not open PDF: {exc}") from exc

    try:
        return "".join(page.get_text() for page in doc)
    finally:
        doc.close()
