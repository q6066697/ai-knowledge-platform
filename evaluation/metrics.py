REFUSAL_MARKERS = [
    "я не нашёл ответа",
    "не знаю",
    "не нашел",
    "не нашёл",
    "недостаточно информации",
    "i don't know",
    "i do not know",
    "i could not find",
    "i couldn't find",
    "no relevant information",
    "not mentioned in the",
    "not contain",
]


def hit_at_k(retrieved_docs: list[str], expected_document: str, k: int) -> bool:
    """True if expected_document appears among the top-k retrieved documents."""
    return expected_document in retrieved_docs[:k]


def mrr(retrieved_docs: list[str], expected_document: str) -> float:
    """Reciprocal rank of the first occurrence of expected_document, 0.0 if absent."""
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc == expected_document:
            return 1.0 / rank
    return 0.0


def answer_correctness(answer_text: str, expected_answer_contains: list[str]) -> bool:
    """True if every expected substring appears in answer_text, case-insensitively."""
    lowered = answer_text.lower()
    return all(substring.lower() in lowered for substring in expected_answer_contains)


def refusal_detected(answer_text: str, sources_empty: bool = False) -> bool:
    """True if the answer looks like a refusal. An empty source list is the
    authoritative signal (the pipeline never called the LLM); otherwise fall
    back to matching known refusal phrases in the answer text."""
    if sources_empty:
        return True
    lowered = answer_text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def hallucination_rate(eval_results: list[dict]) -> float:
    """Fraction of 'unanswerable' questions where refusal_detected is False,
    i.e. the model answered instead of refusing. 0.0 if there are no
    unanswerable questions in eval_results."""
    unanswerable = [r for r in eval_results if r["type"] == "unanswerable"]
    if not unanswerable:
        return 0.0
    hallucinated = sum(1 for r in unanswerable if not r["refusal_detected"])
    return hallucinated / len(unanswerable)
