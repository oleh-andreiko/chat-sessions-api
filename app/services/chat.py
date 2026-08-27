"""Orchestration of one exchange: history -> OpenAI -> cost -> database."""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config
from app.errors import SessionNotFound, UnsupportedModel
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


def active_messages(db: Session, session: ChatSession) -> list[Message]:
    """The session's history since the last reset, oldest first.

    Ordering includes id, not just created_at: the two messages of one exchange
    are written in the same transaction and can share a timestamp, which would
    leave their order — and therefore the order of roles — undefined.
    """
    query = (
        select(Message)
        .where(
            Message.session_id == session.id,
            Message.generation == session.current_generation,
        )
        .order_by(Message.created_at, Message.id)
    )
    return list(db.scalars(query).all())


def _context(db: Session, session: ChatSession) -> list[Message]:
    """The tail of the active history that is replayed to the model."""
    newest_first = (
        select(Message)
        .where(
            Message.session_id == session.id,
            Message.generation == session.current_generation,
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(config.MAX_HISTORY_MESSAGES)
    )
    return list(reversed(db.scalars(newest_first).all()))


def reset_session(db: Session, session_id: uuid.UUID) -> ChatSession:
    """Start a fresh context without touching the session id or its past.

    Earlier messages stay in the table under their old generation. Deleting
    them would throw away the record of money already spent, which is the one
    thing the cost columns exist to preserve.
    """
    session = get_session(db, session_id)
    session.current_generation += 1
    db.commit()
    db.refresh(session)
    return session


def send_message(
    db: Session, session_id: uuid.UUID, content: str, model: str | None = None
) -> Message:
    session = get_session(db, session_id)

    # A model given on the request wins for this message only; the session keeps
    # its own as the default for the next one.
    chosen_model = model or session.model

    # Checked before the API call, not after: an unsupported model would fail at
    # pricing anyway, and by then the request has already been paid for.
    if not pricing.is_supported(chosen_model):
        raise UnsupportedModel(
            f"Model '{chosen_model}' is not supported. "
            f"Supported models: {', '.join(pricing.SUPPORTED_MODELS)}."
        )

    payload = [{"role": m.role, "content": m.content} for m in _context(db, session)]
    payload.append({"role": "user", "content": content})

    try:
        completion = openai_client.complete(chosen_model, payload)
    except Exception:
        logger.exception("OpenAI call failed for session %s", session_id)
        raise

    cost = pricing.calculate_cost(
        chosen_model, completion.prompt_tokens, completion.completion_tokens
    )

    # Both messages are written after the call succeeds, in one transaction.
    # Writing the question first would leave an unanswered message behind on a
    # failed call, and the next request would replay it as context.
    #
    # They are added together on purpose: SQLAlchemy sends both rows in a
    # single INSERT, so their ids stay adjacent even when two requests hit the
    # same session at once. Flushing between the two adds would split that into
    # two statements and let another request's question land in between.
    generation = session.current_generation
    answer = Message(
        session_id=session_id,
        generation=generation,
        role="assistant",
        content=completion.content,
        model=chosen_model,
        resolved_model=completion.resolved_model,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        cost=cost,
    )
    db.add(
        Message(
            session_id=session_id,
            generation=generation,
            role="user",
            content=content,
        )
    )
    db.add(answer)
    db.commit()
    db.refresh(answer)
    return answer
