from openai import OpenAI

from app.core.config import get_settings

settings = get_settings()
_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches of up to EMBEDDING_BATCH_SIZE (default 100) per API call."""
    if not texts:
        return []

    client = get_client()
    batch_size = settings.EMBEDDING_BATCH_SIZE
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=batch)
        embeddings.extend([item.embedding for item in response.data])

    return embeddings
