# ai-knowledge-platform

Production-like RAG-сервис для работы с собственными документами: загружаешь PDF/DOCX/TXT,
задаёшь вопрос — получаешь ответ на основе загруженных документов с указанием источников
(файл, страница, score). В отличие от `rag-eval-service` (бенчмарк на NFCorpus), это
вертикальный срез сервиса с пользовательскими документами: upload → chunking → embeddings →
retrieval → ответ с источниками.

Автор: Alexey Konchalenko ([github.com/q6066697](https://github.com/q6066697))

## Архитектура

```
                 ┌─────────────────┐
                 │  Streamlit UI    │
                 │ (upload, chat)   │
                 └────────┬─────────┘
                          │ HTTP
                          ▼
                 ┌─────────────────┐
                 │   FastAPI app    │
                 │                  │
   upload ──────►│ /documents/upload│──► loaders → chunker → embeddings (OpenAI)
                  │ /documents (GET) │
                  │ /documents/{id}  │──► DELETE (cascade)
                  │       DELETE     │
                  │                  │
   question ─────►│     /chat        │──► similarity_search (pgvector, cosine)
                  │                  │      │
                  │                  │      ├─ best score < threshold → "не знаю", sources: []
                  │                  │      └─ иначе → LLM (gpt-4o-mini) с контекстом → answer + sources
                  └────────┬─────────┘
                           │ SQLAlchemy (async) + asyncpg
                           ▼
                 ┌─────────────────────┐
                 │ PostgreSQL + pgvector│
                 │ documents            │
                 │ document_chunks       │
                 │ (HNSW index, cosine) │
                 └─────────────────────┘
```

## Стек

- Python 3.11+, FastAPI + uvicorn
- PostgreSQL 16 + `pgvector` (Docker Compose, только БД)
- Embeddings: OpenAI `text-embedding-3-small` (1536 dims)
- LLM: OpenAI `gpt-4o-mini` (конфигурируется через `.env`)
- `pypdf`, `python-docx` — парсинг файлов
- SQLAlchemy (async) + `asyncpg`, миграции — Alembic
- Streamlit — UI для ручной проверки
- `pytest` (+ `pytest-asyncio`, `pytest-mock`)
- `pydantic-settings` — конфиг через `.env`

## Как запустить локально

```bash
# 1. Поднять Postgres + pgvector
docker compose up -d

# 2. Установить зависимости
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Настроить .env
cp .env.example .env
# вписать свой OPENAI_API_KEY в .env вручную

# 4. Применить миграции
alembic upgrade head

# 5. Запустить API
uvicorn app.main:app --reload

# 6. Запустить UI (отдельный терминал)
streamlit run frontend/streamlit_app.py
```

API — `http://localhost:8000` (Swagger: `/docs`), UI — `http://localhost:8501`.

## Пример запроса/ответа

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@docs/sample-data/test-knowledge-base.pdf"
# {"id": "...", "filename": "test-knowledge-base.pdf", "status": "completed", "chunks_created": 3}

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days do I have to return a device?"}'
```

```json
{
  "answer": "Вы имеете 45 дней с момента первоначальной покупки, чтобы вернуть устройство Nimbus Cloud Storage.",
  "sources": [
    {"document_id": "...", "filename": "test-knowledge-base.pdf", "page": 1, "chunk_index": 0, "score": 0.496}
  ]
}
```

Скриншоты полного цикла (upload → список документов → чат с источниками) — в `docs/screenshots/`.

## Дизайн-решения

**Postgres + pgvector, а не отдельная векторная БД.** Для V0.1 не нужна отдельная
инфраструктура (Pinecone/Qdrant/Weaviate) — один Postgres хранит и метаданные документов, и
эмбеддинги, с транзакционной консистентностью (upload и запись чанков — в одной БД). HNSW-индекс
даёт приемлемую скорость ANN-поиска для одного пользователя/небольшого корпуса. Если объём
данных вырастет — миграция на специализированную векторную БД возможна без изменения бизнес-логики
(интерфейс `similarity_search` не завязан на конкретную БД).

**Постраничный chunking с overlap, а не фиксированный посимвольный сплит.** Фиксированная
нарезка по N символов режет предложения и абзацы посередине, что портит качество retrieval —
эмбеддинг обрубленного куска смысла хуже отражает содержание. Сплит по абзацам с последующим
объединением до `CHUNK_SIZE` и overlap между соседними чанками сохраняет смысловую целостность и
даёт котекст на границах чанков. Page-level tracking позволяет указывать пользователю точную
страницу источника.

**Anti-hallucination guardrail: порог similarity + системный промпт (defense in depth).**
Первый уровень защиты — если лучший результат поиска ниже `SIMILARITY_THRESHOLD`, LLM не
вызывается вовсе (экономия, детерминированный "не знаю"). Второй уровень — даже когда чанки
прошли порог, но не содержат прямого ответа, системный промпт инструктирует модель явно
признать нехватку информации, а не придумывать факт (проверено на скриншотах — модель отвечает
"Не знаю" на вопрос без ответа в контексте, несмотря на то что похожие по теме чанки были
найдены).

## Roadmap / что дальше

Сознательно не сделано в V0.1 (вертикальный срез, а не полная архитектура):

- **V0.2**: асинхронная обработка документов через Celery/Redis (сейчас — синхронно в запросе)
- Hybrid search (dense + BM25 + RRF), reranking — как в `rag-eval-service`
- Evaluation harness (аналог BEIR-эвала из `rag-eval-service`) для количественной оценки retrieval/answer quality
- Langfuse — трейсинг LLM-вызовов и стоимости
- Telegram-бот поверх `/chat`
- MCP-сервер поверх этого API (по аналогии с `rag-mcp-server`)
- Docker для самого приложения, CI/CD

## Безопасность

`OPENAI_API_KEY` и `DATABASE_URL` — только через `.env` (в `.gitignore`, не коммитится).
`.env.example` — с пустыми плейсхолдерами.
