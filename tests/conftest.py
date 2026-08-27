"""Shared test setup.

Tests run against a real PostgreSQL — the same server the app uses, but a
separate database. SQLite would be simpler, yet the schema leans on Postgres
types (UUID columns, numeric precision), so testing on SQLite would prove the
code works somewhere it never runs.

DATABASE_URL is rewritten to the test database *before* the app is imported,
because app.config reads it once at import time. Everything downstream — the
engine, the app, the dependency — then points at the test database on its own,
with no override needed.
"""

import os
from decimal import Decimal

import psycopg
import pytest
from dotenv import load_dotenv
from sqlalchemy.engine import make_url

# app.config loads .env too, but that happens on import — and the test database
# URL has to be in place before then.
load_dotenv()

TEST_DB_NAME = "app_test"


def _prepare_test_database() -> str:
    """Create the test database if it is not there yet, return its URL."""
    dev_url = make_url(os.environ["DATABASE_URL"])
    test_url = dev_url.set(database=TEST_DB_NAME)

    # CREATE DATABASE cannot run inside a transaction, hence autocommit, and it
    # cannot run from the database being created, hence connecting to "postgres".
    admin = psycopg.connect(
        host=dev_url.host,
        port=dev_url.port,
        user=dev_url.username,
        password=dev_url.password,
        dbname="postgres",
        autocommit=True,
    )
    with admin:
        exists = admin.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB_NAME,)
        ).fetchone()
        if not exists:
            admin.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')

    return test_url.render_as_string(hide_password=False)


os.environ["DATABASE_URL"] = _prepare_test_database()
# A key must be present for the app to import; no test ever calls OpenAI.
os.environ.setdefault("OPENAI_API_KEY", "test-key-never-used")

from fastapi.testclient import TestClient  # noqa: E402

from app.db import SessionLocal, engine  # noqa: E402
from app.errors import UpstreamError  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Base, ChatSession, Message  # noqa: E402
from app.services import chat, openai_client  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _schema():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture(autouse=True)
def _clean_tables():
    """Every test starts on empty tables, so none can depend on another."""
    yield
    with SessionLocal() as db:
        db.query(Message).delete()
        db.query(ChatSession).delete()
        db.commit()


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def fake_openai(monkeypatch):
    """Replace the OpenAI call and record what context it was given.

    Nothing here is about OpenAI's behaviour: what these tests check is which
    messages we decided to send, and what we do with the usage that comes back.
    """
    calls = []

    def complete(model, messages):
        calls.append({"model": model, "messages": messages})
        return openai_client.Completion(
            content=f"reply {len(calls)}",
            prompt_tokens=100,
            completion_tokens=10,
            resolved_model=f"{model}-2026-01-01",
        )

    monkeypatch.setattr(chat.openai_client, "complete", complete)
    return calls


@pytest.fixture
def session_id(client):
    return client.post("/sessions", json={"title": "test"}).json()["id"]


@pytest.fixture
def failing_openai(monkeypatch):
    """Make the OpenAI call raise the way a real failure does."""

    def boom(model, messages):
        raise UpstreamError("Upstream model request failed.")

    monkeypatch.setattr(chat.openai_client, "complete", boom)
