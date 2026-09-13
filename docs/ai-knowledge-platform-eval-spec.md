# Тех-спек для Claude Code: ai-knowledge-platform — Evaluation Harness

> Работай в существующем репозитории `ai-knowledge-platform`. Скопируй как промпт для Claude Code. Иди по разделу 8 "Порядок работы" пошагово, показывай реальный вывод на каждом шаге.

## 0. Контекст и отличие от rag-eval-service

В `rag-eval-service` eval был построен на **статичном бенчмарке** (NFCorpus/BEIR, ~3600 документов, готовые qrels) — это оценка retrieval-компонентов в изоляции на "чужом" датасете. Здесь задача другая: оценить **весь live-сервис end-to-end** (upload → retrieval → LLM-ответ) на **своих документах** — маленький, но осмысленный golden set, привязанный к реальному контенту `docs/sample-data/test-knowledge-base.pdf` (уже есть в репозитории). Это осознанно другой тип eval — не бенчмарк для резюме-цифр, а регрессионный harness для самого продукта: "не сломали ли мы retrieval/anti-hallucination при следующих изменениях".

Отдельная ценность этого harness (которой не было в rag-eval-service): у `/chat` уже есть флаги `ENABLE_HYBRID_SEARCH` / `ENABLE_RERANKING` — значит можно **сравнить 3 конфигурации retrieval на одном и том же датасете и вопросах** (dense-only vs hybrid vs hybrid+rerank) без переписывания кода под каждый вариант, просто дергая internal-функцию с разными флагами.

## 1. Golden set

Файл `evaluation/golden_set.jsonl`, каждая строка:
```json
{
  "id": "q1",
  "question": "How many days do I have to return a device?",
  "type": "answerable",
  "expected_answer_contains": ["45"],
  "expected_source_document": "test-knowledge-base.pdf",
  "expected_source_page": 1
}
```

Для неотвечаемых вопросов:
```json
{
  "id": "q_neg1",
  "question": "Does Nimbus support Android?",
  "type": "unanswerable",
  "expected_refusal": true
}
```

Составить **минимум 12 вопросов**:
- 6 "answerable" — по 3 на каждую страницу `test-knowledge-base.pdf` (return policy, warranty, pricing, support — уже известные тебе факты из предыдущих ручных тестов), с разной лексикой (часть вопросов — точные термины из текста для проверки lexical/hybrid-ветки, часть — перефразированные синонимами для проверки dense-ветки)
- 6 "unanswerable" — вопросы не по теме документа (аналогично ручным тестам: "CEO name", "Android support" и т.п., плюс пара вопросов, которые звучат похоже на тему документа, но спрашивают то, чего там реально нет — например, про функцию, не упомянутую в тексте)

## 2. Метрики (evaluation/metrics.py)

Retrieval-метрики (аналогично rag-eval-service, но на своём golden set):
- `hit_at_k(retrieved_docs, expected_document, k)` 
- `mrr(retrieved_docs, expected_document)`

Answer-level метрики (новое, специфичное для этого harness):
- `answer_correctness(answer_text, expected_answer_contains)` — простая keyword-проверка (все ожидаемые подстроки/числа присутствуют в ответе, регистронезависимо)
- `refusal_detected(answer_text)` — эвристика на отказ (ответ похож на "не знаю"/"не нашёл" — можно захардкодить несколько маркеров-фраз или проверить, что sources пуст, как более надёжный сигнал, чем парсинг текста ответа)
- `hallucination_rate` — доля "unanswerable" вопросов, где `refusal_detected == False` (то есть модель выдумала ответ вместо отказа) — это ключевая метрика для anti-hallucination нарратива

## 3. Runner (evaluation/run_eval.py)

- НЕ ходить через живой HTTP `/chat` (это требует перезапуска сервера с разными env для каждой конфигурации). Вместо этого вызывать внутреннюю функцию оркестрации (`retrieval/pipeline.py` + `llm/client.py`) напрямую из скрипта, передавая `enable_hybrid` / `enable_reranking` как параметры функции (если сейчас они читаются только из `Settings` — добавь возможность передать override явным аргументом, не трогая поведение самого `/chat` эндпоинта).
- Прогнать golden set через 3 конфигурации: `dense_only`, `hybrid_no_rerank`, `hybrid_rerank`.
- Вывести таблицу (markdown, для вставки в README):

```
| Config           | Hit@3 | MRR  | Answer correctness | Hallucination rate |
|------------------|-------|------|---------------------|---------------------|
| dense_only       | ...   | ...  | ...                 | ...                 |
| hybrid_no_rerank | ...   | ...  | ...                 | ...                 |
| hybrid_rerank    | ...   | ...  | ...                 | ...                 |
```

- Сохранить сырые результаты (per-question) в `evaluation/results/latest.json` — для дебага, что именно пошло не так на конкретном вопросе, если метрика упала.

## 4. Тесты

- `test_metrics.py` — unit-тесты на каждую метрику (`hit_at_k`, `mrr`, `answer_correctness`, `refusal_detected`) на синтетических примерах, без реальных вызовов LLM/БД.
- Не гонять полный eval-прогон (с реальными OpenAI-вызовами) в обычном `pytest` suite по умолчанию — это стоит денег и времени при каждом CI-прогоне. Вынести `run_eval.py` как отдельный ручной скрипт (`python -m evaluation.run_eval`), а не pytest-тест. Явно зафиксировать это решение в README ("почему eval — не часть pytest suite").

## 5. README — что добавить

- Новый раздел "Evaluation" с:
  - объяснением, чем этот harness отличается от рагnostic-бенчмарка в rag-eval-service (см. п.0)
  - таблицей результатов 3 конфигураций
  - интерпретацией: если `hybrid_rerank` не даёт видимого прироста на 12 вопросах — это честно объяснить (маленький golden set, другой характер данных, чем NFCorpus) вместо того, чтобы подгонять нарратив под "reranking всегда лучше"
  - explicit `hallucination_rate` числом — это самая наглядная метрика для интервью
- Обновить Roadmap — убрать "evaluation harness" из нереализованного.

## 6. Что НЕ делать

- Не переносить golden_set/датасет из rag-eval-service — это разные датасеты для разных целей.
- Не добавлять LLM-as-judge оценку (более сложная и дорогая метрика) — простой keyword-based correctness достаточно для V0.1 этого harness; можно упомянуть как возможное развитие в README.
- Не встраивать запуск eval в CI/pre-commit — только ручной запуск.

## 7. Ручная проверка (тебе)

После реализации — прогони `python -m evaluation.run_eval` сам и посмотри на `hallucination_rate` для каждой конфигурации: если у `dense_only` он выше, чем у `hybrid_rerank` — это будет хорошим количественным доказательством того, что доп. слои retrieval не просто "красивая архитектура", а реально снижают галлюцинации. Если разницы нет — это тоже честный и интересный результат, не нужно его прятать.

## 8. Порядок работы

1. `golden_set.jsonl` (12 вопросов, вручную проверить, что они соответствуют реальному содержимому test-knowledge-base.pdf).
2. `metrics.py` + `test_metrics.py`, прогнать тесты.
3. Добавить override-параметры в pipeline (`enable_hybrid`, `enable_reranking` как явные аргументы функции, не только из Settings) — не ломая текущее поведение `/chat`.
4. `run_eval.py`, прогнать один раз вручную, показать полную таблицу и `results/latest.json`.
5. README — секция Evaluation с таблицей и интерпретацией.
6. `git status`, коммит, push.
