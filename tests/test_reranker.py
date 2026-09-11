import uuid

from app.core.config import get_settings
from app.retrieval.reranker import rerank
from app.retrieval.vector_search import SearchResult

settings = get_settings()


def _result(content: str, chunk_index: int) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="test.txt",
        content=content,
        page_number=1,
        chunk_index=chunk_index,
        score=0.0,
    )


def test_rerank_reorders_candidates_relative_to_input_order():
    """Input order deliberately puts an irrelevant chunk first (as RRF might,
    e.g. from a lexical false-positive) and the truly relevant chunk last.
    The cross-encoder reads (query, passage) together and should promote the
    relevant chunk above its RRF input rank."""
    query = "What is the capital of France?"
    candidates = [
        _result("Bananas are a good source of potassium.", 0),
        _result("The stock market closed lower on Tuesday.", 1),
        _result("Paris is the capital and most populous city of France.", 2),
    ]

    reranked = rerank(query, candidates, settings.RERANKER_MODEL, top_k=3)

    assert [r.chunk_index for r in reranked] != [c.chunk_index for c in candidates]
    assert reranked[0].chunk_index == 2


def test_rerank_respects_top_k():
    query = "What is the capital of France?"
    candidates = [
        _result("Paris is the capital of France.", 0),
        _result("Berlin is the capital of Germany.", 1),
        _result("Bananas are yellow.", 2),
    ]

    reranked = rerank(query, candidates, settings.RERANKER_MODEL, top_k=1)

    assert len(reranked) == 1


def test_rerank_empty_candidates_returns_empty():
    reranked = rerank("anything", [], settings.RERANKER_MODEL, top_k=5)
    assert reranked == []
