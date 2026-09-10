# Тех-спек для Claude Code: ai-knowledge-platform (V0.1 MVP)

> Скопируй этот файл целиком как первый промпт для Claude Code в новой пустой директории проекта. Работай с Claude Code пошагово: после каждого крупного блока (БД → ingestion → RAG → API → тесты) проси его показать вывод команды/теста, а не просто "готово" — так же, как ты проверял rag-eval-service.

## 0. Контекст для агента

Это новый портфолио-репозиторий Alex (Alexey Konchalenko), AI/Python фрилансера. Уже есть в портфолио:
- `langgraph-research-agent` — ReAct-агент на LangGraph с веб-поиском и цитированием
- `rag-eval-service` — гибридный поиск (dense+BM25+RRF) + reranking + eval на NFCorpus (BEIR)
- `rag-mcp-server` — MCP-сервер поверх rag-eval-service

`ai-knowledge-platform` — следующая ступень: production-like RAG-сервис с документами пользователя (upload → ответ с источниками), а не бенчмарк-проект. Цель V0.1 — рабочий вертикальный срез (vertical slice), а не полная архитектура из readme-плана. Все продвинутые части (hybrid search, reranking, Celery, Langfuse, Telegram, MCP) — НЕ делать сейчас, они пойдут отдельными итерациями в следующих сессиях.

## 1. Название и репозиторий

- Имя репозитория: `ai-knowledge-platform`
- Автор в README и git config: Alexey Konchalenko, GitHub github.com/q6066697

## 2. Стек (только для V0.1)

- Python 3.11+, FastAPI + uvicorn
- PostgreSQL 16 + расширение `pgvector` (через `docker-compose.yml`, только для БД — само приложение пока запускается локально `uvicorn`, без Docker для app)
- Embeddings: OpenAI `text-embedding-3-small` (1536 dims) — тот же выбор, что в rag-eval-service, для консистентности
- LLM для ответов: OpenAI (модель — `gpt-4o-mini` по умолчанию, вынести в конфиг)
- Парсинг файлов: `pypdf` (PDF), `python-docx` (DOCX), обычный `open()` для TXT
- ORM/доступ к БД: `SQLAlchemy` + `asyncpg`, миграции — `alembic`
- UI для ручной проверки: `Streamlit` (отдельный процесс, ходит в FastAPI по HTTP)
- Тесты: `pytest`
- Конфиг: `pydantic-settings`, секреты — через `.env` (НЕ коммитить, добавить `.env.example`)

## 3. Структура репозитория

```
ai-knowledge-platform/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── documents.py       # upload/list/delete
│   │   └── chat.py            # POST /chat
│   ├── core/
│   │   ├── config.py          # pydantic-settings: DATABASE_URL, OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
│   │   └── logging.py
│   ├── db/
│   │   ├── database.py        # async engine/session
│   │   └── models.py          # Document, DocumentChunk (embedding: Vector(1536))
│   ├── ingestion/
│   │   ├── loaders.py         # extract_text(file, file_type) -> list[(page_number, text)]
│   │   ├── chunker.py         # chunk_text(pages, chunk_size, overlap) -> list[Chunk]
│   │   └── embeddings.py      # embed_texts(list[str]) -> list[list[float]]
│   ├── retrieval/
│   │   └── vector_search.py   # similarity_search(query_embedding, top_k) -> list[chunk+score]
│   ├── llm/
│   │   └── client.py          # answer_with_context(question, chunks) -> answer text
│   └── schemas/
│       ├── documents.py       # Pydantic request/response models
│       └── chat.py
├── frontend/
│   └── streamlit_app.py
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_chat.py
├── alembic/                   # migrations
├── docker-compose.yml         # только postgres+pgvector
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 4. Схема БД

```sql
documents
  id            uuid primary key default gen_random_uuid()
  filename      text not null
  file_type     text not null           -- pdf | docx | txt
  status        text not null default 'processing'  -- processing | completed | failed
  created_at    timestamptz not null default now()

document_chunks
  id             uuid primary key default gen_random_uuid()
  document_id    uuid not null references documents(id) on delete cascade
  content        text not null
  page_number    int
  chunk_index    int not null
  embedding      vector(1536) not null
  created_at     timestamptz not null default now()
```

Индекс для поиска: `CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);` (или `ivfflat`, если Claude Code сочтёт проще для V0.1 — оба варианта приемлемы, зафиксировать выбор в README).

## 5. Ingestion pipeline — требования

1. `extract_text` возвращает текст **постранично** (список `(page_number, text)`), чтобы позже можно было указать источник с номером страницы. Для TXT — вся страница = 1.
2. Chunking — **не фиксированный** посимвольный сплит без учёта смысла. Минимально: сплит по абзацам (`\n\n`), затем объединение соседних абзацев до `CHUNK_SIZE` (по умолчанию 700 символов) с overlap (по умолчанию 100 символов), не разрывая абзац посередине без необходимости. Сохранить `page_number` и `chunk_index` для каждого чанка.
3. Эмбеддинги — батчами (не по одному чанку за вызов), максимум ~100 чанков за один запрос к OpenAI API.
4. Статус документа: `processing` при загрузке → `completed` после успешной обработки → `failed` при ошибке (с логом причины).
5. Обработка **синхронная** внутри запроса `POST /documents/upload` для V0.1 (без очередей — это сознательное упрощение, вынести в README как "V0.2: async processing via Celery").

## 6. API endpoints

```
POST   /documents/upload      multipart/form-data, поле "file" (pdf/docx/txt)
                               → {"id": uuid, "filename": str, "status": "completed", "chunks_created": int}

GET    /documents             → [{"id", "filename", "file_type", "status", "created_at"}]

DELETE /documents/{id}        → 204, каскадно удаляет chunks

POST   /chat
  body: {"question": str, "top_k": int optional}
  → {
      "answer": str,
      "sources": [
        {"document_id": uuid, "filename": str, "page": int, "chunk_index": int, "score": float}
      ]
    }

GET    /health                → {"status": "ok"}
```

## 7. RAG-логика и защита от галлюцинаций

1. Эмбеддинг вопроса → `similarity_search` (cosine) → top_k чанков (по умолчанию 5).
2. Если релевантных чанков нет или лучший score ниже порога (`SIMILARITY_THRESHOLD`, вынести в конфиг, начальное значение подобрать эмпирически) — **не звать LLM**, сразу вернуть `answer: "Я не нашёл ответа в загруженных документах."`, `sources: []`.
3. System-промпт для LLM (зафиксировать в коде примерно так, можно уточнить формулировки):

```
Ты — ассистент базы знаний. Отвечай ТОЛЬКО на основе предоставленного контекста.
Если в контексте недостаточно информации для ответа — прямо скажи, что не знаешь.
Не придумывай факты, которых нет в контексте.
Отвечай на языке вопроса пользователя.
```

4. В `sources` — только реально использованные чанки (те, что были переданы в контекст), с `document_id`, `filename`, `page`, `score`.

## 8. Streamlit UI (frontend/streamlit_app.py)

Минимально:
- Форма загрузки файла → вызывает `POST /documents/upload`
- Список загруженных документов с статусами (`GET /documents`), кнопка удаления
- Поле вопроса → `POST /chat` → показать ответ и список источников (файл + страница + score)

## 9. Тесты (pytest)

- `test_chunking.py`: chunker не разрывает короткий абзац; overlap применяется корректно; edge case — текст короче chunk_size.
- `test_retrieval.py`: similarity_search возвращает чанки, отсортированные по убыванию score (можно с мок-эмбеддингами/in-memory векторами, без реального OpenAI-вызова).
- `test_chat.py`: при пустой БД `/chat` возвращает "не нашёл ответа" и пустой `sources`, а не 500 и не выдуманный ответ.

Тесты, которые бьют в реальный OpenAI API, — замокать (`unittest.mock` / `pytest-mock`), чтобы CI не требовал ключ.

## 10. README.md — обязательные разделы

1. Краткое описание проекта (2-3 предложения, что это и зачем)
2. Архитектурная схема (ASCII или ссылка на картинку)
3. Стек
4. Как запустить локально (docker-compose up для БД, alembic upgrade, uvicorn, streamlit run)
5. Пример запроса/ответа `/chat` (curl или httpie)
6. Дизайн-решения, которые стоит объяснить на собеседовании:
   - почему postgres+pgvector, а не отдельная векторная БД
   - почему постраничный chunking с overlap, а не фиксированный посимвольный
   - anti-hallucination guardrail (порог схожести + системный промпт)
7. Явный раздел "Roadmap / что дальше" со списком V0.2+ (hybrid search, reranking, Celery, evaluation harness, Langfuse, Telegram, MCP) — честно обозначить как следующие итерации, а не как недоделанное

## 11. Безопасность

- `OPENAI_API_KEY` и `DATABASE_URL` — только через `.env`, который добавить в `.gitignore`. `.env.example` — с пустыми плейсхолдерами.
- **Не вписывай реальный API-ключ через Claude Code** — впиши его в `.env` вручную сам, после того как Claude Code создаст `.env.example` (та же практика, что в rag-mcp-server).

## 12. Явно НЕ делать в V0.1 (чтобы не расползался скоуп)

- Celery/Redis, hybrid search (BM25), reranking, evaluation harness, Langfuse-трейсинг, Telegram-бот, MCP-сервер, Docker для самого приложения (только для БД), CI/CD.

## 13. Порядок работы

1. Инициализация репозитория, структура папок, `requirements.txt`, `.gitignore`, `docker-compose.yml` (postgres+pgvector).
2. Модели + alembic-миграция, проверить `alembic upgrade head` реально создаёт таблицы (показать вывод).
3. `ingestion/` (loaders → chunker → embeddings), покрыть `test_chunking.py`, прогнать и показать результат.
4. `retrieval/vector_search.py` + `test_retrieval.py`.
5. `POST /documents/upload`, `GET /documents`, `DELETE /documents/{id}` — вручную проверить через curl на реальном тестовом PDF.
6. `POST /chat` + anti-hallucination логика + `test_chat.py`.
7. Streamlit UI, ручная сквозная проверка (upload → question → answer + sources).
8. README, финальная проверка `git status` на отсутствие `.env` в коммите, push в GitHub.
