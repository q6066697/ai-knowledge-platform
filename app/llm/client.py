from openai import OpenAI

from app.core.config import get_settings
from app.retrieval.vector_search import SearchResult

settings = get_settings()
_client: OpenAI | None = None

SYSTEM_PROMPT = """Ты — ассистент базы знаний. Отвечай ТОЛЬКО на основе предоставленного контекста.
Если в контексте недостаточно информации для ответа — прямо скажи, что не знаешь.
Не придумывай факты, которых нет в контексте.
Отвечай на языке вопроса пользователя."""


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def _build_context(chunks: list[SearchResult]) -> str:
    parts = []
    for chunk in chunks:
        label = f"[{chunk.filename}, стр. {chunk.page_number}]" if chunk.page_number else f"[{chunk.filename}]"
        parts.append(f"{label}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def answer_with_context(question: str, chunks: list[SearchResult]) -> str:
    context = _build_context(chunks)
    client = get_client()

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
        ],
    )
    return response.choices[0].message.content or ""
