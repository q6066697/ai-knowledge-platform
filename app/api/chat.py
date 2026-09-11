from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.database import get_db
from app.llm.client import answer_with_context
from app.retrieval.pipeline import retrieve
from app.schemas.chat import ChatRequest, ChatResponse, SourceItem

settings = get_settings()
router = APIRouter(tags=["chat"])

NO_ANSWER_TEXT = "Я не нашёл ответа в загруженных документах."


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)) -> ChatResponse:
    top_k = request.top_k or settings.TOP_K

    results = await retrieve(db, request.question, top_k=top_k)

    if not results or results[0].score < settings.SIMILARITY_THRESHOLD:
        return ChatResponse(answer=NO_ANSWER_TEXT, sources=[])

    answer = answer_with_context(request.question, results)

    sources = [
        SourceItem(
            document_id=r.document_id,
            filename=r.filename,
            page=r.page_number,
            chunk_index=r.chunk_index,
            score=r.score,
        )
        for r in results
    ]

    return ChatResponse(answer=answer, sources=sources)
