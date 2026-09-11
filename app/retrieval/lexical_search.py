from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk
from app.retrieval.vector_search import SearchResult


async def lexical_search(session: AsyncSession, query: str, top_k: int = 5) -> list[SearchResult]:
    """Full-text search over document_chunks via Postgres tsvector/GIN index.
    Returns chunks matching at least one query term, ranked by ts_rank_cd
    descending. English text search config, matching content_tsv."""
    tsquery = func.plainto_tsquery("english", query)
    rank = func.ts_rank_cd(DocumentChunk.content_tsv, tsquery)

    stmt = (
        select(DocumentChunk, Document.filename, rank.label("rank"))
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(DocumentChunk.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
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
            score=float(rank_value),
        )
        for chunk, filename, rank_value in rows
    ]
