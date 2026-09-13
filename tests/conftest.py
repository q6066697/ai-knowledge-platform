import os
from pathlib import Path

from dotenv import dotenv_values

# Tests must never run against the same database as a manually-run app
# (Streamlit/uvicorn) — test_chat.py's `empty_database` fixture deletes all
# documents in whatever DB it connects to. This overrides DATABASE_URL to
# TEST_DATABASE_URL *before* any app module is imported (app.db.database
# creates its engine from settings.DATABASE_URL at import time), so the
# whole test session — including TestClient-driven /chat and /documents
# tests — runs against the isolated test database.
_repo_root = Path(__file__).resolve().parent.parent
_dotenv = dotenv_values(_repo_root / ".env")

test_database_url = os.environ.get("TEST_DATABASE_URL") or _dotenv.get("TEST_DATABASE_URL")

if not test_database_url:
    raise RuntimeError(
        "TEST_DATABASE_URL is not set (checked environment and .env). "
        "Tests must run against a separate database from the app — "
        "see .env.example and the 'Running tests' section in README.md."
    )

os.environ["DATABASE_URL"] = test_database_url
