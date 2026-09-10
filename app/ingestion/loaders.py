from pathlib import Path
from typing import BinaryIO

from docx import Document as DocxDocument
from pypdf import PdfReader


def extract_text(file: str | Path | BinaryIO, file_type: str) -> list[tuple[int, str]]:
    """Extract text page by page. Returns a list of (page_number, text), 1-indexed."""
    file_type = file_type.lower().lstrip(".")

    if file_type == "pdf":
        reader = PdfReader(file)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]

    if file_type == "docx":
        doc = DocxDocument(file)
        text = "\n\n".join(p.text for p in doc.paragraphs)
        return [(1, text)]

    if file_type == "txt":
        if hasattr(file, "read"):
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
        else:
            content = Path(file).read_text(encoding="utf-8")
        return [(1, content)]

    raise ValueError(f"Unsupported file_type: {file_type}")
