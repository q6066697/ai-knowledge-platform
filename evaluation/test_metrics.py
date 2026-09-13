from evaluation.metrics import (
    answer_correctness,
    hallucination_rate,
    hit_at_k,
    mrr,
    refusal_detected,
)


def test_hit_at_k_true_when_expected_document_within_top_k():
    assert hit_at_k(["a.pdf", "b.pdf", "c.pdf"], "b.pdf", k=3) is True


def test_hit_at_k_false_when_expected_document_outside_top_k():
    assert hit_at_k(["a.pdf", "b.pdf", "c.pdf"], "c.pdf", k=2) is False


def test_hit_at_k_false_when_expected_document_absent():
    assert hit_at_k(["a.pdf", "b.pdf"], "z.pdf", k=5) is False


def test_hit_at_k_handles_empty_retrieved_list():
    assert hit_at_k([], "a.pdf", k=3) is False


def test_mrr_is_one_when_expected_document_is_first():
    assert mrr(["a.pdf", "b.pdf"], "a.pdf") == 1.0


def test_mrr_is_reciprocal_of_rank_when_not_first():
    assert mrr(["a.pdf", "b.pdf", "c.pdf"], "c.pdf") == 1.0 / 3


def test_mrr_is_zero_when_expected_document_absent():
    assert mrr(["a.pdf", "b.pdf"], "z.pdf") == 0.0


def test_answer_correctness_true_when_all_substrings_present():
    assert answer_correctness("You have 45 days to return it.", ["45", "return"]) is True


def test_answer_correctness_false_when_one_substring_missing():
    assert answer_correctness("You have 45 days to return it.", ["45", "warranty"]) is False


def test_answer_correctness_is_case_insensitive():
    assert answer_correctness("Refund window is 45 DAYS.", ["45 days"]) is True


def test_answer_correctness_true_for_empty_expected_list():
    assert answer_correctness("anything at all", []) is True


def test_refusal_detected_true_when_sources_empty():
    assert refusal_detected("some fabricated answer", sources_empty=True) is True


def test_refusal_detected_true_for_known_refusal_phrase():
    assert refusal_detected("Я не нашёл ответа в загруженных документах.") is True


def test_refusal_detected_true_for_english_refusal_phrase():
    assert refusal_detected("I don't know based on the provided context.") is True


def test_refusal_detected_false_for_confident_answer():
    assert refusal_detected("The subscription costs 19.99 USD per month.") is False


def test_hallucination_rate_is_zero_when_all_unanswerable_refused():
    results = [
        {"type": "unanswerable", "refusal_detected": True},
        {"type": "unanswerable", "refusal_detected": True},
        {"type": "answerable", "refusal_detected": False},
    ]
    assert hallucination_rate(results) == 0.0


def test_hallucination_rate_counts_only_unanswerable_that_were_not_refused():
    results = [
        {"type": "unanswerable", "refusal_detected": False},
        {"type": "unanswerable", "refusal_detected": True},
        {"type": "unanswerable", "refusal_detected": False},
        {"type": "answerable", "refusal_detected": False},
    ]
    assert hallucination_rate(results) == 2 / 3


def test_hallucination_rate_is_zero_when_no_unanswerable_questions():
    results = [{"type": "answerable", "refusal_detected": False}]
    assert hallucination_rate(results) == 0.0
