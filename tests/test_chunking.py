from app.ingestion.chunker import chunk_text


def test_text_shorter_than_chunk_size_produces_single_chunk():
    pages = [(1, "Короткий текст, который весь помещается в один чанк.")]

    chunks = chunk_text(pages, chunk_size=700, overlap=100)

    assert len(chunks) == 1
    assert chunks[0].content == pages[0][1]
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0


def test_short_paragraph_is_not_split():
    paragraph = "Это короткий абзац."
    pages = [(1, paragraph)]

    chunks = chunk_text(pages, chunk_size=700, overlap=100)

    assert len(chunks) == 1
    assert chunks[0].content == paragraph


def test_paragraphs_are_merged_until_chunk_size():
    paragraphs = ["A" * 300, "B" * 300, "C" * 300]
    text = "\n\n".join(paragraphs)
    pages = [(1, text)]

    chunks = chunk_text(pages, chunk_size=700, overlap=100)

    assert len(chunks) == 2
    assert "A" * 300 in chunks[0].content
    assert "B" * 300 in chunks[0].content
    assert "C" * 300 not in chunks[0].content
    assert "C" * 300 in chunks[1].content


def test_overlap_carries_tail_of_previous_chunk_into_next():
    paragraphs = ["A" * 300, "B" * 300, "C" * 300]
    text = "\n\n".join(paragraphs)
    pages = [(1, text)]
    overlap = 100

    chunks = chunk_text(pages, chunk_size=700, overlap=overlap)

    assert len(chunks) == 2
    tail_of_first = chunks[0].content[-overlap:]
    assert tail_of_first in chunks[1].content


def test_paragraph_longer_than_chunk_size_is_force_split():
    long_paragraph = "X" * 1500
    pages = [(1, long_paragraph)]

    chunks = chunk_text(pages, chunk_size=700, overlap=100)

    assert len(chunks) >= 2
    assert all(len(c.content) <= 700 for c in chunks)
    # First chunk starts with no overlap and matches the first raw slice exactly.
    assert chunks[0].content == long_paragraph[:700]


def test_chunk_index_and_page_number_are_preserved_across_pages():
    pages = [(1, "Первая страница."), (2, "Вторая страница.")]

    chunks = chunk_text(pages, chunk_size=700, overlap=100)

    assert [c.page_number for c in chunks] == [1, 2]
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_empty_pages_produce_no_chunks():
    pages = [(1, ""), (2, "   ")]

    chunks = chunk_text(pages, chunk_size=700, overlap=100)

    assert chunks == []
