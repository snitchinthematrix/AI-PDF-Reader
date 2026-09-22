from pdf_reader.text_utils import preprocess_text, split_into_chunks


def test_preprocess_text_collapses_whitespace():
    assert preprocess_text("Hello   \n\n world") == "Hello world"


def test_preprocess_text_strips_special_characters():
    assert preprocess_text("Price: $5.00 (approx)!") == "Price 5.00 approx!"


def test_preprocess_text_keeps_basic_sentence_punctuation():
    assert preprocess_text("Wait, really? Yes - absolutely!") == "Wait, really? Yes - absolutely!"


def test_split_into_chunks_empty_text_returns_no_chunks():
    assert split_into_chunks("") == []


def test_split_into_chunks_single_chunk_when_short():
    chunks = split_into_chunks("Only one short sentence.", chunk_size=500)
    assert chunks == ["Only one short sentence."]


def test_split_into_chunks_respects_chunk_size():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = split_into_chunks(text, chunk_size=20)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_split_into_chunks_each_chunk_ends_on_a_sentence():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = split_into_chunks(text, chunk_size=20)
    assert all(chunk.endswith(".") for chunk in chunks)
