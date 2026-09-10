from dataclasses import dataclass


@dataclass
class Chunk:
    content: str
    page_number: int | None
    chunk_index: int


def _split_paragraphs(text: str, chunk_size: int) -> list[str]:
    """Split into paragraphs on blank lines; force-split any paragraph longer than chunk_size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    normalized: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            normalized.append(paragraph)
        else:
            for i in range(0, len(paragraph), chunk_size):
                normalized.append(paragraph[i : i + chunk_size])

    return normalized


def chunk_text(
    pages: list[tuple[int, str]], chunk_size: int = 700, overlap: int = 100
) -> list[Chunk]:
    """Chunk page text by paragraph, merging adjacent paragraphs up to chunk_size,
    carrying an overlap tail from the previous chunk into the next one."""
    chunks: list[Chunk] = []
    chunk_index = 0

    for page_number, page_text in pages:
        paragraphs = _split_paragraphs(page_text, chunk_size)
        current = ""

        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}" if current else paragraph

            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(Chunk(content=current, page_number=page_number, chunk_index=chunk_index))
                chunk_index += 1
                tail = current[-overlap:] if overlap > 0 else ""
                with_tail = f"{tail}\n\n{paragraph}" if tail else paragraph
                current = with_tail if len(with_tail) <= chunk_size else paragraph
            else:
                current = paragraph

        if current:
            chunks.append(Chunk(content=current, page_number=page_number, chunk_index=chunk_index))
            chunk_index += 1

    return chunks
