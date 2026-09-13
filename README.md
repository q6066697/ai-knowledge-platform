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
   question ─────►│     /chat        │──► retrieval/pipeline.py:
                  │                  │      │
                  │                  │      ├─ similarity_search (pgvector, cosine, top-50)
                  │                  │      ├─ lexical_search (Postgres FTS, ts_rank_cd, top-50)
                  │                  │      ├─ reciprocal_rank_fusion (RRF, k=60)
                  │                  │      ├─ rerank (cross-encoder ms-marco-MiniLM-L-6-v2, top-10 → top-K)
                  │                  │      ├─ best rerank score < SIMILARITY_THRESHOLD → "не знаю", sources: []
                  │                  │      └─ иначе → LLM (gpt-4o-mini) с контекстом → answer + sources
                  └────────┬─────────┘
                           │ SQLAlchemy (async) + asyncpg
                           ▼
                 ┌──────────────────────────────┐
                 │ PostgreSQL + pgvector        │
                 │  documents                   │
                 │  document_chunks             │
                 │   - embedding (HNSW, cosine) │
                 │   - content_tsv (GIN, FTS)   │
                 └──────────────────────────────┘
```

`ENABLE_HYBRID_SEARCH=false` and/or `ENABLE_RERANKING=false` (см. `.env.example`) откатывают пайплайн к чистому dense-поиску V0.1 без изменения кода.

## Стек

- Python 3.11+, FastAPI + uvicorn
- PostgreSQL 16 + `pgvector` (Docker Compose, только БД)
- Embeddings: OpenAI `text-embedding-3-small` (1536 dims)
- LLM: OpenAI `gpt-4o-mini` (конфигурируется через `.env`)
- Lexical retrieval: встроенный full-text search PostgreSQL (`tsvector` + GIN, `ts_rank_cd`)
- Reranking: `sentence-transformers` cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU)
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

## Тесты

Тесты используют **отдельную** базу данных, а не ту, что использует запущенное приложение —
`test_chat.py::empty_database` удаляет все документы/чанки в базе, к которой подключается, и
её нельзя случайно направить на базу с реальными данными.

```bash
# 1. Добавить TEST_DATABASE_URL в .env (отдельная БД, не DATABASE_URL!)
#    см. .env.example — по умолчанию ai_knowledge_platform_test
echo "TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ai_knowledge_platform_test" >> .env

# 2. Создать тестовую БД и прогнать миграции
docker exec ai-knowledge-platform-db createdb -U postgres ai_knowledge_platform_test
DATABASE_URL=$TEST_DATABASE_URL alembic upgrade head

# 3. Запустить тесты
pytest -v
```

`tests/conftest.py` переопределяет `DATABASE_URL` на `TEST_DATABASE_URL` до импорта любых
модулей приложения (движок SQLAlchemy создаётся при импорте `app.db.database`), поэтому весь
прогон — включая `TestClient`-тесты `/chat` и `/documents` — идёт через тестовую БД. Если
`TEST_DATABASE_URL` не задан, `conftest.py` явно падает с ошибкой вместо того, чтобы тихо
использовать `DATABASE_URL`.

Проверено вручную: `GET /documents` на запущенном (не тестовом) приложении даёт одинаковый
результат до и после полного прогона `pytest`, при этом счётчик документов в
`ai_knowledge_platform_test` после прогона — 0.

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
    {"document_id": "...", "filename": "test-knowledge-base.pdf", "page": 1, "chunk_index": 0, "score": 4.834}
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

## V0.2: Hybrid Search + Reranking

V0.1 использовал только dense retrieval (pgvector cosine). V0.2 добавляет lexical retrieval и
reranking поверх него, по аналогии с тем, что было измерено в портфолио-проекте
`rag-eval-service` на BEIR/NFCorpus (naive hybrid не всегда бьёт dense, а reranking даёт
устойчивый прирост +8% ndcg@10/mrr). Здесь — не бенчмарк, а встраивание той же техники в
реальный сервис с живыми документами пользователя.

**Пайплайн**: dense (pgvector, top-50) + lexical (Postgres FTS, top-50) → RRF (k=60) →
cross-encoder rerank (top-10 кандидатов → top-K) → threshold-check на финальном rerank-score →
LLM. Реализация — `app/retrieval/{lexical_search,fusion,reranker,pipeline}.py`.

**Lexical retrieval через Postgres FTS, а не `rank-bm25`.** В `rag-eval-service` лексический
поиск был через `rank-bm25` на статичном in-memory корпусе — это работает, когда корпус не
меняется. Здесь корпус динамический (документы загружаются/удаляются через API), и
пересобирать in-memory BM25-индекс при каждом upload/delete — лишняя сложность и лишнее
состояние. Вместо этого — генерируемая колонка `content_tsv tsvector` (`to_tsvector('english',
content)`, `GENERATED ALWAYS ... STORED`) с GIN-индексом: Postgres сам поддерживает её в
актуальном состоянии на каждый insert/update, без отдельного индекса приложения. Ограничение:
конфиг full-text search зафиксирован на `english` — датасет пока англоязычный.

**RRF (Reciprocal Rank Fusion), k=60.** Чанк на позиции `r` (с 1) в списке получает
`1/(k+r)`; вклады из dense- и lexical-списков суммируются. Фьюжн происходит по **рангам**, а
не по сырым score — cosine similarity и `ts_rank_cd` несравнимы по шкале напрямую. `k=60` —
то же значение, что в `rag-eval-service` (сознательная консистентность для сравнения между
проектами, не re-tuned под этот корпус).

**Reranking отдельным слоем поверх RRF.** В `rag-eval-service` было явно показано: naive
RRF-фьюжн не гарантирует улучшения качества сам по себе, а cross-encoder reranking даёт
надёжный прирост релевантности, потому что читает `(query, passage)` вместе, а не как два
независимо посчитанных вектора. Тот же вывод применён здесь: RRF формирует кандидатный пул
(top-10 после фьюжна), а `cross-encoder/ms-marco-MiniLM-L-6-v2` его переранжирует перед
передачей в LLM.

**Подбор `SIMILARITY_THRESHOLD` под rerank-шкалу.** Cross-encoder возвращает сырой logit
(не cosine similarity 0..1), поэтому старое значение порога (`0.3`) с V0.1 не переносится.
Порог подобран прогоном пайплайна на `test-knowledge-base.pdf` по вопросам трёх типов:

| Тип вопроса | Пример | Top-1 rerank score |
|---|---|---|
| Релевантный, факт есть в документе | "How many days do I have to return a device?" | **+4.83** |
| Релевантный, факт есть в документе | "How much does the Nimbus subscription cost per month?" | **+9.49** |
| Тема релевантна, факта нет (guardrail-кейс) | "Does Nimbus support Android?" | **-2.18** |
| Полностью нерелевантный | "What is the capital of France?" | **-11.15** |
| Полностью нерелевантный | "How do I bake a chocolate cake?" | **-11.08** |

Явный разрыв между отвечаемыми вопросами (+4.8…+9.5) и guardrail-кейсом (-2.18) позволил
взять **`SIMILARITY_THRESHOLD = 0.0`** — круглое число внутри разрыва, не подогнанное под
конкретные наблюдения. Anti-hallucination guardrail из V0.1 (см. ниже) продолжает работать на
новой шкале: тема близка ("Does Nimbus support Android?" — про тот же продукт), но факта нет,
и rerank-score всё равно уходит в отрицательную область.

**Независимое переключение слоёв.** `ENABLE_HYBRID_SEARCH` и `ENABLE_RERANKING` (`.env.example`,
default `true` для обоих) позволяют откатиться к чистому dense-поиску V0.1 без изменения кода —
для сравнения/дебага и как демонстрация "могу включить/выключить каждый слой отдельно".

## Roadmap / что дальше

Сознательно не сделано в V0.2 (вертикальный срез, а не полная архитектура):

- Асинхронная обработка документов через Celery/Redis (сейчас — синхронно в запросе)
- Evaluation harness (аналог BEIR-эвала из `rag-eval-service`) для количественной оценки retrieval/answer quality
- Langfuse — трейсинг LLM-вызовов и стоимости
- Telegram-бот поверх `/chat`
- MCP-сервер поверх этого API (по аналогии с `rag-mcp-server`)
- Docker для самого приложения, CI/CD

## Безопасность

`OPENAI_API_KEY` и `DATABASE_URL` — только через `.env` (в `.gitignore`, не коммитится).
`.env.example` — с пустыми плейсхолдерами.
