from sentence_transformers import CrossEncoder

from app.retrieval.vector_search import SearchResult

_model: CrossEncoder | None = None
_model_name: str | None = None


def get_reranker(model_name: str) -> CrossEncoder:
    global _model, _model_name
    if _model is None or _model_name != model_name:
        _model = CrossEncoder(model_name)
        _model_name = model_name
    return _model


def rerank(
    query: str, candidates: list[SearchResult], model_name: str, top_k: int
) -> list[SearchResult]:
    """Cross-encoder reranking: reads (query, chunk) together for a more
    accurate relevance score than RRF's rank-only fusion. Returns the
    top_k candidates sorted by rerank score descending, with `score`
    replaced by the cross-encoder's raw output (a different scale than
    cosine similarity or RRF)."""
    if not candidates:
        return []

    model = get_reranker(model_name)
    pairs = [(query, c.content) for c in candidates]
    scores = model.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)[:top_k]

    return [
        SearchResult(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            filename=c.filename,
            content=c.content,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            score=float(score),
        )
        for c, score in ranked
    ]
