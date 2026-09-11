import pytest
import pytest_asyncio
from sqlalchemy import delete

from app.db.database import AsyncSessionLocal
from app.db.models import Document, DocumentChunk
from app.retrieval.lexical_search import lexical_search

pytestmark = pytest.mark.asyncio


def _embedding(seed: float) -> list[float]:
    """Deterministic fake 1536-dim embedding, unrelated to lexical content on
    purpose: these tests check term matching, not vector proximity."""
    vec = [0.0] * 1536
    vec[0] = seed
    return vec


@pytest_asyncio.fixture
async def sample_document():
    async with AsyncSessionLocal() as session:
        document = Document(filename="test.txt", file_type="txt", status="completed")
        session.add(document)
        await session.flush()

        chunks = [
            DocumentChunk(
                document_id=document.id,
                content="The warranty covers product model ZX-9982 for one year.",
                page_number=1,
                chunk_index=0,
                embedding=_embedding(0.1),
            ),
            DocumentChunk(
                document_id=document.id,
                content="General terms and conditions apply to all purchases.",
                page_number=1,
                chunk_index=1,
                embedding=_embedding(0.2),
            ),
            DocumentChunk(
                document_id=document.id,
                content="Refunds are processed within 45 days of the original purchase.",
                page_number=2,
                chunk_index=2,
                embedding=_embedding(0.3),
            ),
        ]
        session.add_all(chunks)
        await session.commit()
        document_id = document.id

    yield document_id

    async with AsyncSessionLocal() as session:
        await session.execute(delete(Document).where(Document.id == document_id))
        await session.commit()


async def test_lexical_search_finds_exact_term_not_close_by_embedding(sample_document):
    """ZX-9982 is a rare product code: its fake embedding (seed 0.1) is not
    close to any plausible query embedding, so only lexical match finds it."""
    async with AsyncSessionLocal() as session:
        results = await lexical_search(session, "ZX-9982", top_k=10)

    own_results = [r for r in results if r.document_id == sample_document]
    assert len(own_results) == 1
    assert "ZX-9982" in own_results[0].content


async def test_lexical_search_respects_top_k(sample_document):
    async with AsyncSessionLocal() as session:
        results = await lexical_search(session, "purchase", top_k=1)

    assert len(results) <= 1


async def test_lexical_search_orders_by_rank_descending(sample_document):
    async with AsyncSessionLocal() as session:
        results = await lexical_search(session, "purchase", top_k=10)

    own_results = [r for r in results if r.document_id == sample_document]
    assert len(own_results) >= 1
    scores = [r.score for r in own_results]
    assert scores == sorted(scores, reverse=True)


async def test_lexical_search_returns_empty_for_no_term_match(sample_document):
    async with AsyncSessionLocal() as session:
        results = await lexical_search(session, "xyzxyzxyz_nonexistent_term", top_k=10)

    own_results = [r for r in results if r.document_id == sample_document]
    assert own_results == []
