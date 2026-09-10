import io
import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.database import get_db
from app.db.models import Document, DocumentChunk
from app.ingestion.chunker import chunk_text
from app.ingestion.embeddings import embed_texts
from app.ingestion.loaders import extract_text
from app.schemas.documents import DocumentResponse, DocumentUploadResponse

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_FILE_TYPES = {"pdf", "docx", "txt"}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
) -> DocumentUploadResponse:
    file_type = ""
    if file.filename and "." in file.filename:
        file_type = file.filename.rsplit(".", 1)[-1].lower()

    if file_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type or 'unknown'}")

    content = await file.read()

    document = Document(filename=file.filename, file_type=file_type, status="processing")
    db.add(document)
    await db.commit()
    await db.refresh(document)

    try:
        pages = extract_text(io.BytesIO(content), file_type)
        chunks = chunk_text(pages, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)

        if chunks:
            chunk_embeddings = embed_texts([c.content for c in chunks])
            db.add_all(
                DocumentChunk(
                    document_id=document.id,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    embedding=embedding,
                )
                for chunk, embedding in zip(chunks, chunk_embeddings)
            )

        document.status = "completed"
        await db.commit()

        return DocumentUploadResponse(
            id=document.id,
            filename=document.filename,
            status=document.status,
            chunks_created=len(chunks),
        )
    except Exception:
        logger.exception("Failed to process document %s", document.id)
        document.status = "failed"
        await db.commit()
        return DocumentUploadResponse(
            id=document.id,
            filename=document.filename,
            status=document.status,
            chunks_created=0,
        )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: AsyncSession = Depends(get_db)) -> list[DocumentResponse]:
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    return list(result.scalars().all())


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> None:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    await db.execute(delete(Document).where(Document.id == document_id))
    await db.commit()
