"""Manual evaluation harness for the live RAG pipeline (upload -> retrieval -> LLM answer).

NOT a pytest test: it makes real OpenAI calls (embeddings + LLM) against the
app's actual DATABASE_URL, so it costs money and takes time. Run it by hand:

    python -m evaluation.run_eval

It expects docs/sample-data/test-knowledge-base.pdf to already be uploaded to
the running app's database (see README "Как запустить локально").
"""

import asyncio
import json
import statistics
from pathlib import Path
from typing import Any

from app.api.chat import NO_ANSWER_TEXT
from app.core.config import get_settings
from app.db.database import AsyncSessionLocal, engine
from app.llm.client import answer_with_context
from app.retrieval.pipeline import retrieve
from evaluation.metrics import answer_correctness, hallucination_rate, hit_at_k, mrr, refusal_detected

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
RESULTS_PATH = Path(__file__).parent / "results" / "latest.json"
HIT_AT_K = 3

CONFIGS: dict[str, dict[str, bool]] = {
    "dense_only": {"enable_hybrid": False, "enable_reranking": False},
    "hybrid_no_rerank": {"enable_hybrid": True, "enable_reranking": False},
    "hybrid_rerank": {"enable_hybrid": True, "enable_reranking": True},
}

settings = get_settings()


def load_golden_set() -> list[dict[str, Any]]:
    with open(GOLDEN_SET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


async def evaluate_question(question: dict[str, Any], enable_hybrid: bool, enable_reranking: bool) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        results = await retrieve(
            session,
            question["question"],
            top_k=settings.TOP_K,
            enable_hybrid=enable_hybrid,
            enable_reranking=enable_reranking,
        )

    # Mirrors app/api/chat.py's anti-hallucination guardrail exactly, so the
    # eval reflects real /chat behavior rather than always calling the LLM.
    if not results or results[0].score < settings.SIMILARITY_THRESHOLD:
        answer = NO_ANSWER_TEXT
        sources_empty = True
    else:
        answer = answer_with_context(question["question"], results)
        sources_empty = False

    retrieved_docs = [r.filename for r in results]
    entry: dict[str, Any] = {
        "id": question["id"],
        "question": question["question"],
        "type": question["type"],
        "retrieved_docs": retrieved_docs,
        "answer": answer,
        "sources_empty": sources_empty,
        "refusal_detected": refusal_detected(answer, sources_empty=sources_empty),
    }

    if question["type"] == "answerable":
        expected_document = question["expected_source_document"]
        entry["hit_at_3"] = hit_at_k(retrieved_docs, expected_document, k=HIT_AT_K)
        entry["reciprocal_rank"] = mrr(retrieved_docs, expected_document)
        entry["answer_correct"] = answer_correctness(answer, question["expected_answer_contains"])
    else:
        entry["hit_at_3"] = None
        entry["reciprocal_rank"] = None
        entry["answer_correct"] = None

    return entry


async def evaluate_config(
    config_name: str, enable_hybrid: bool, enable_reranking: bool, golden_set: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_question = [await evaluate_question(q, enable_hybrid, enable_reranking) for q in golden_set]

    answerable = [r for r in per_question if r["type"] == "answerable"]
    summary = {
        "config": config_name,
        "hit_at_3": statistics.mean(r["hit_at_3"] for r in answerable),
        "mrr": statistics.mean(r["reciprocal_rank"] for r in answerable),
        "answer_correctness": statistics.mean(r["answer_correct"] for r in answerable),
        "hallucination_rate": hallucination_rate(per_question),
    }
    return summary, per_question


def render_markdown_table(summaries: list[dict[str, Any]]) -> str:
    header = "| Config           | Hit@3 | MRR  | Answer correctness | Hallucination rate |"
    separator = "|------------------|-------|------|---------------------|---------------------|"
    rows = [
        f"| {s['config']:<16} | {s['hit_at_3']:.2f}  | {s['mrr']:.2f} | "
        f"{s['answer_correctness']:.2f}                | {s['hallucination_rate']:.2f}                |"
        for s in summaries
    ]
    return "\n".join([header, separator, *rows])


async def main() -> None:
    golden_set = load_golden_set()
    summaries = []
    all_per_question = {}

    for config_name, flags in CONFIGS.items():
        print(f"Running config: {config_name} ...")
        summary, per_question = await evaluate_config(config_name, flags["enable_hybrid"], flags["enable_reranking"], golden_set)
        summaries.append(summary)
        all_per_question[config_name] = per_question

    table = render_markdown_table(summaries)
    print("\n" + table + "\n")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"summaries": summaries, "per_question": all_per_question}, f, indent=2, ensure_ascii=False)
    print(f"Raw per-question results saved to {RESULTS_PATH}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
