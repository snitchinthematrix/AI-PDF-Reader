# 📄 AI PDF Reader

A local, offline PDF reading and analysis tool. Upload a PDF, get an extractive
summary and sentiment read on any section, and ask free-form questions about
the text — all with models that run entirely on your own CPU. No API keys, no
data leaves your machine.

Built with [Streamlit](https://streamlit.io/), [PyMuPDF](https://pymupdf.readthedocs.io/),
and local [Hugging Face](https://huggingface.co/) / [Sentence-Transformers](https://www.sbert.net/)
models.

## Features

- **PDF text extraction** — pulls text from any standard PDF via PyMuPDF.
- **Chunking** — splits long documents into readable, sentence-aligned chunks.
- **Sentiment analysis** — DistilBERT (SST-2) sentiment on the selected chunk.
- **Key-point extraction** — ranks sentences in a chunk by cosine similarity to
  the chunk's overall embedding (MiniLM sentence embeddings) and surfaces the
  most representative ones.
- **Question answering** — extractive QA (DistilBERT/SQuAD) over the selected
  chunk.

## Architecture

```
app.py                    Streamlit UI: wiring, caching, user interaction
pdf_reader/
  text_utils.py            Pure text cleaning/chunking (stdlib only)
  pdf_extraction.py         PDF -> text (PyMuPDF)
  analyzer.py               Sentiment / key points / QA (HF + sentence-transformers)
tests/
  test_text_utils.py        Unit tests for chunking/cleaning
  test_analyzer.py          Unit tests for analyzer logic, via injected fakes
```

The three concerns — PDF parsing, text processing, and ML inference — are
separated into their own modules so each can be tested and reasoned about
independently of Streamlit and of each other.

## Getting started

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the three local models (a few hundred MB total) from
Hugging Face; subsequent runs use the local cache.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The unit tests inject lightweight fakes for the sentiment/embedding/QA models
(see `DocumentAnalyzer`'s constructor in `pdf_reader/analyzer.py`), so the full
suite runs in under a second with just `numpy` and `pytest` installed —
no multi-gigabyte model download required to verify the logic.

GitHub Actions runs `ruff check .` and `pytest -v` on every push and pull
request (see `.github/workflows/ci.yml`), for the same reason: no heavy ML
dependencies needed in CI.

## Notable design decisions

- **Model loading is cached, not repeated.** The original version constructed
  a fresh `LocalPDFReader()` — reloading all three models — on every Streamlit
  rerun (i.e. every button click). Model loading now happens once per session
  via `st.cache_resource`, and PDF extraction/chunking is cached per file via
  `st.cache_data`.
- **Dependency injection over mocking frameworks.** `DocumentAnalyzer` accepts
  its three models as constructor arguments, defaulting to the real ones. This
  keeps `pdf_reader/analyzer.py` importable, and its ranking/fallback logic
  testable, without torch/transformers installed at all.
- **numpy over torch for similarity scoring.** Key-point ranking uses a small
  numpy cosine-similarity helper instead of converting embeddings to torch
  tensors, which was unnecessary overhead for a CPU-bound similarity lookup.
- **Errors surface to the user, not a stack trace.** A corrupt/unreadable PDF
  or a scanned/image-only PDF (no extractable text) now shows a clear
  in-app message instead of crashing the app.

## Known limitations / possible next steps

- No OCR fallback for scanned/image-only PDFs.
- Key-point ranking is a simple extractive heuristic (embedding similarity to
  the whole chunk), not an abstractive summary.
- Single-file upload only; no multi-document comparison.
- No persistence — analysis is re-run per session.

## License

MIT — see [LICENSE](LICENSE).
