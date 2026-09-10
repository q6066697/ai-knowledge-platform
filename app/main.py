from fastapi import FastAPI

from app.api import chat, documents
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(title="ai-knowledge-platform")

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
