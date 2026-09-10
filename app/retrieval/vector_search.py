import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk


@dataclass
class SearchResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    page_number: int | None
    chunk_index: int
    score: float


async def similarity_search(
    session: AsyncSession, query_embedding: list[float], top_k: int = 5
) -> list[SearchResult]:
    """Cosine similarity search over document_chunks. Returns results sorted by
    score (1 - cosine_distance) descending, most similar first."""
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)

    stmt = (
        select(DocumentChunk, Document.filename, distance.label("distance"))
        .join(Document, DocumentChunk.document_id == Document.id)
        .order_by(distance)
        .limit(top_k)
    )

    rows = (await session.execute(stmt)).all()

    return [
        SearchResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=filename,
            content=chunk.content,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            score=1.0 - distance_value,
        )
        for chunk, filename, distance_value in rows
    ]
