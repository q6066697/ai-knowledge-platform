from app.retrieval.vector_search import SearchResult


def _key(r: SearchResult) -> tuple:
    return (r.document_id, r.chunk_index)


def reciprocal_rank_fusion(
    dense_results: list[SearchResult], lexical_results: list[SearchResult], k: int = 60
) -> list[SearchResult]:
    """Fuse dense and lexical result lists by RANK, not raw score — the two
    scales (cosine similarity vs. ts_rank_cd) aren't comparable. A chunk at
    rank r (1-indexed) in a list contributes 1/(k+r); contributions from both
    lists are summed. Returns chunks sorted by fused score descending, with
    `score` on each SearchResult replaced by its fused RRF score."""
    fused_scores: dict[tuple, float] = {}
    by_key: dict[tuple, SearchResult] = {}

    for results in (dense_results, lexical_results):
        for rank, r in enumerate(results, start=1):
            key = _key(r)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            by_key.setdefault(key, r)

    ordered_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)

    fused = []
    for key in ordered_keys:
        original = by_key[key]
        fused.append(
            SearchResult(
                chunk_id=original.chunk_id,
                document_id=original.document_id,
                filename=original.filename,
                content=original.content,
                page_number=original.page_number,
                chunk_index=original.chunk_index,
                score=fused_scores[key],
            )
        )
    return fused
