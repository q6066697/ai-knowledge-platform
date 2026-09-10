import pytest
import pytest_asyncio
from sqlalchemy import delete

from app.db.database import AsyncSessionLocal
from app.db.models import Document, DocumentChunk
from app.retrieval.vector_search import similarity_search

pytestmark = pytest.mark.asyncio


def _embedding(seed: float) -> list[float]:
    """Deterministic fake 1536-dim embedding, no real OpenAI call involved."""
    vec = [0.0] * 1536
    vec[0] = seed
    vec[1] = 1.0 - seed
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
                content=f"chunk {i}",
                page_number=1,
                chunk_index=i,
                embedding=_embedding(seed),
            )
            for i, seed in enumerate([0.1, 0.9, 0.5])
        ]
        session.add_all(chunks)
        await session.commit()
        document_id = document.id

    yield document_id

    async with AsyncSessionLocal() as session:
        await session.execute(delete(Document).where(Document.id == document_id))
        await session.commit()


async def test_similarity_search_orders_results_by_score_descending(sample_document):
    query_embedding = _embedding(0.9)  # should match "chunk 1" (seed 0.9) exactly

    async with AsyncSessionLocal() as session:
        results = await similarity_search(session, query_embedding, top_k=10)

    own_results = [r for r in results if r.document_id == sample_document]
    assert len(own_results) == 3

    scores = [r.score for r in own_results]
    assert scores == sorted(scores, reverse=True)
    assert own_results[0].content == "chunk 1"
    assert own_results[0].score == pytest.approx(1.0, abs=1e-6)


async def test_similarity_search_respects_top_k(sample_document):
    query_embedding = _embedding(0.9)

    async with AsyncSessionLocal() as session:
        results = await similarity_search(session, query_embedding, top_k=1)

    assert len(results) == 1
    assert results[0].content == "chunk 1"


async def test_similarity_search_includes_filename_and_metadata(sample_document):
    query_embedding = _embedding(0.5)

    async with AsyncSessionLocal() as session:
        results = await similarity_search(session, query_embedding, top_k=10)

    own_result = next(r for r in results if r.document_id == sample_document and r.content == "chunk 2")
    assert own_result.filename == "test.txt"
    assert own_result.page_number == 1
    assert own_result.chunk_index == 2
