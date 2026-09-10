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
    SIMILARITY_THRESHOLD: float = 0.3

    EMBEDDING_BATCH_SIZE: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
