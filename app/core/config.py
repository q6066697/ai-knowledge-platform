from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DATABASE_URL: str
    OPENAI_API_KEY: str = ""

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536
    LLM_MODEL: str = "gpt-4o-mini"

    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.0

    EMBEDDING_BATCH_SIZE: int = 100

    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_RERANKING: bool = True
    RRF_K: int = 60
    RERANK_CANDIDATES: int = 10
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache
def get_settings() -> Settings:
    return Settings()
