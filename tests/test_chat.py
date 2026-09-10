import pytest
from fastapi.testclient import TestClient

from app.main import app

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
    mocker.patch("app.api.chat.embed_texts", return_value=[[0.0] * 1536])

    response = client.post("/chat", json={"question": "What is in the documents?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == NO_ANSWER_TEXT
    assert data["sources"] == []


def test_chat_does_not_call_llm_when_no_relevant_chunks(client, mocker):
    mock_embed = mocker.patch("app.api.chat.embed_texts", return_value=[[0.0] * 1536])
    mock_answer = mocker.patch("app.api.chat.answer_with_context")

    response = client.post("/chat", json={"question": "Anything?"})

    assert response.status_code == 200
    mock_embed.assert_called_once()
    mock_answer.assert_not_called()
