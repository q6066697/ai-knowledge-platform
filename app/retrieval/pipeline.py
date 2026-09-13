from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.ingestion.embeddings import embed_texts
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.lexical_search import lexical_search
from app.retrieval.reranker import rerank
from app.retrieval.vector_search import SearchResult, similarity_search

settings = get_settings()

CANDIDATE_POOL_SIZE = 50


async def retrieve(
    session: AsyncSession,
    question: str,
    top_k: int,
    enable_hybrid: bool | None = None,
    enable_reranking: bool | None = None,
) -> list[SearchResult]:
    """Orchestrates dense -> (lexical -> RRF) -> rerank. ENABLE_HYBRID_SEARCH
    and ENABLE_RERANKING gate each layer independently, so either can be
    turned off to fall back toward V0.1 dense-only retrieval without code
    changes. When both are off, this reproduces V0.1's similarity_search
    call exactly (same top_k, no lexical/rerank calls).

    enable_hybrid/enable_reranking default to None, which falls back to the
    Settings flags used by /chat — passing an explicit bool overrides them
    for a single call (used by evaluation/run_eval.py to compare
    configurations without restarting the server)."""
    hybrid = settings.ENABLE_HYBRID_SEARCH if enable_hybrid is None else enable_hybrid
    reranking = settings.ENABLE_RERANKING if enable_reranking is None else enable_reranking

    [query_embedding] = embed_texts([question])

    if hybrid:
        dense_results = await similarity_search(session, query_embedding, top_k=CANDIDATE_POOL_SIZE)
        lexical_results = await lexical_search(session, question, top_k=CANDIDATE_POOL_SIZE)
        candidates = reciprocal_rank_fusion(dense_results, lexical_results, k=settings.RRF_K)
    else:
        pool_size = settings.RERANK_CANDIDATES if reranking else top_k
        candidates = await similarity_search(session, query_embedding, top_k=max(pool_size, top_k))

    if reranking:
        return rerank(question, candidates[: settings.RERANK_CANDIDATES], settings.RERANKER_MODEL, top_k=top_k)

    return candidates[:top_k]
