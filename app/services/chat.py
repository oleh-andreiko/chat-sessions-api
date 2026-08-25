"""Orchestration of one exchange: history -> OpenAI -> cost -> database."""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config
from app.errors import SessionNotFound
from app.models import ChatSession, Message
from app.services import openai_client, pricing

logger = logging.getLogger(__name__)


def get_session(db: Session, session_id: uuid.UUID) -> ChatSession:
    session = db.get(ChatSession, session_id)
    if session is None:
        raise SessionNotFound(f"Session {session_id} does not exist.")
    return session


def create_session(db: Session, model: str | None, title: str | None) -> ChatSession:
    session = ChatSession(model=model or config.DEFAULT_MODEL, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _history(db: Session, session_id: uuid.UUID) -> list[Message]:
    """Last MAX_HISTORY_MESSAGES messages of the session, oldest first.

    Ordering includes id, not just created_at: the two messages of one exchange
    are written in the same transaction and can share a timestamp, which would
    leave their order — and therefore the order of roles — undefined.
    """
    newest_first = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(config.MAX_HISTORY_MESSAGES)
    )
    return list(reversed(db.scalars(newest_first).all()))


def send_message(db: Session, session_id: uuid.UUID, content: str) -> Message:
    session = get_session(db, session_id)

    payload = [{"role": m.role, "content": m.content} for m in _history(db, session_id)]
    payload.append({"role": "user", "content": content})

    try:
        completion = openai_client.complete(session.model, payload)
    except Exception:
        logger.exception("OpenAI call failed for session %s", session_id)
        raise

    cost = pricing.calculate_cost(
        session.model, completion.prompt_tokens, completion.completion_tokens
    )

    # Both messages are written after the call succeeds, in one transaction.
    # Writing the question first would leave an unanswered message behind on a
    # failed call, and the next request would replay it as context.
    answer = Message(
        session_id=session_id,
        role="assistant",
        content=completion.content,
        model=session.model,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        cost=cost,
    )
    db.add(Message(session_id=session_id, role="user", content=content))
    db.add(answer)
    db.commit()
    db.refresh(answer)
    return answer
