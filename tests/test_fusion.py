import uuid

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.vector_search import SearchResult


def _result(chunk_index: int, document_id: uuid.UUID, score: float = 0.0) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=document_id,
        filename="test.txt",
        content=f"chunk {chunk_index}",
        page_number=1,
        chunk_index=chunk_index,
        score=score,
    )


def test_chunk_in_both_lists_outranks_chunk_in_one_list_at_same_rank():
    doc_id = uuid.uuid4()
    chunk_a = _result(0, doc_id)  # present in both lists
    chunk_b = _result(1, doc_id)  # present only in dense, same rank as A

    dense_results = [chunk_a, chunk_b]  # both rank 1 and 2 respectively
    lexical_results = [_result(2, doc_id), _result(3, doc_id), chunk_a]  # chunk_a at rank 3

    fused = reciprocal_rank_fusion(dense_results, lexical_results, k=60)

    fused_by_index = {r.chunk_index: r.score for r in fused}
    assert fused_by_index[0] > fused_by_index[1]


def test_fused_results_sorted_by_score_descending():
    doc_id = uuid.uuid4()
    dense_results = [_result(0, doc_id), _result(1, doc_id), _result(2, doc_id)]
    lexical_results = [_result(2, doc_id), _result(0, doc_id)]

    fused = reciprocal_rank_fusion(dense_results, lexical_results, k=60)

    scores = [r.score for r in fused]
    assert scores == sorted(scores, reverse=True)


def test_chunk_only_in_dense_still_included():
    doc_id = uuid.uuid4()
    dense_results = [_result(0, doc_id)]
    lexical_results: list[SearchResult] = []

    fused = reciprocal_rank_fusion(dense_results, lexical_results, k=60)

    assert len(fused) == 1
    assert fused[0].chunk_index == 0
    assert fused[0].score == 1.0 / 61


def test_rrf_score_formula_matches_reciprocal_rank_sum():
    doc_id = uuid.uuid4()
    chunk = _result(0, doc_id)

    dense_results = [chunk]  # rank 1
    lexical_results = [_result(9, doc_id), chunk]  # rank 2

    fused = reciprocal_rank_fusion(dense_results, lexical_results, k=60)

    fused_chunk = next(r for r in fused if r.chunk_index == 0)
    expected = 1.0 / (60 + 1) + 1.0 / (60 + 2)
    assert fused_chunk.score == expected
