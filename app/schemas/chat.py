import uuid

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceItem(BaseModel):
    document_id: uuid.UUID
    filename: str
    page: int | None
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
