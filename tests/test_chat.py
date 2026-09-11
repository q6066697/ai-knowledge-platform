import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.retrieval import pipeline as pipeline_module

NO_ANSWER_TEXT = "Я не нашёл ответа в загруженных документах."


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def empty_database(client):
    """Ensure /chat is tested against an empty documents table, without
    depending on leftover state from manual testing or other test files."""
    for doc in client.get("/documents").json():
        client.delete(f"/documents/{doc['id']}")
    yield


def test_chat_with_empty_database_returns_no_answer_and_no_sources(client, mocker):
    mocker.patch("app.retrieval.pipeline.embed_texts", return_value=[[0.0] * 1536])

    response = client.post("/chat", json={"question": "What is in the documents?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == NO_ANSWER_TEXT
    assert data["sources"] == []


def test_chat_does_not_call_llm_when_no_relevant_chunks(client, mocker):
    mock_embed = mocker.patch("app.retrieval.pipeline.embed_texts", return_value=[[0.0] * 1536])
    mock_answer = mocker.patch("app.api.chat.answer_with_context")

    response = client.post("/chat", json={"question": "Anything?"})

    assert response.status_code == 200
    mock_embed.assert_called_once()
    mock_answer.assert_not_called()


def test_chat_regression_v01_dense_only_when_hybrid_and_reranking_disabled(client, mocker):
    """ENABLE_HYBRID_SEARCH=false + ENABLE_RERANKING=false must reproduce
    V0.1 behavior exactly: a single similarity_search(top_k) call, no
    lexical search, no reranking — confirms the new layers don't change the
    baseline path when switched off."""
    mocker.patch.object(pipeline_module.settings, "ENABLE_HYBRID_SEARCH", False)
    mocker.patch.object(pipeline_module.settings, "ENABLE_RERANKING", False)

    mock_embed = mocker.patch("app.retrieval.pipeline.embed_texts", return_value=[[0.0] * 1536])
    mock_similarity = mocker.patch("app.retrieval.pipeline.similarity_search", return_value=[])
    mock_lexical = mocker.patch("app.retrieval.pipeline.lexical_search")
    mock_rerank = mocker.patch("app.retrieval.pipeline.rerank")

    response = client.post("/chat", json={"question": "Anything?"})

    assert response.status_code == 200
    mock_embed.assert_called_once()
    mock_similarity.assert_called_once()
    assert mock_similarity.call_args.kwargs["top_k"] == pipeline_module.settings.TOP_K
    mock_lexical.assert_not_called()
    mock_rerank.assert_not_called()
