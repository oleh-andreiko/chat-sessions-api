"""HTTP layer: validate, call a service, return a status code. No logic here."""

import uuid
from decimal import Decimal

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import schemas
from app.db import get_db
from app.models import Message
from app.services import chat

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=schemas.SessionOut, status_code=201)
def create_session(payload: schemas.SessionCreate, db: Session = Depends(get_db)):
    return chat.create_session(db, payload.model, payload.title)


@router.post("/{session_id}/messages", response_model=schemas.MessageReply)
def send_message(
    session_id: uuid.UUID,
    payload: schemas.MessageCreate,
    db: Session = Depends(get_db),
):
    answer = chat.send_message(db, session_id, payload.content, payload.model)
    return schemas.MessageReply(
        message=schemas.MessageOut.model_validate(answer),
        model=answer.model,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        cost=answer.cost,
    )


def _detail(db: Session, session) -> schemas.SessionDetail:
    """Session plus the history and totals of its current generation.

    Totals are summed on read instead of being kept as counters on the session
    row: no denormalised value can drift out of sync this way, and after a reset
    there is no counter to remember to zero.
    """
    messages = chat.active_messages(db, session)
    return schemas.SessionDetail(
        id=session.id,
        title=session.title,
        model=session.model,
        current_generation=session.current_generation,
        created_at=session.created_at,
        messages=[schemas.MessageOut.model_validate(m) for m in messages],
        total_prompt_tokens=sum(m.prompt_tokens or 0 for m in messages),
        total_completion_tokens=sum(m.completion_tokens or 0 for m in messages),
        total_cost=sum((m.cost or Decimal(0) for m in messages), Decimal(0)),
    )


@router.get("/{session_id}", response_model=schemas.SessionDetail)
def get_session(session_id: uuid.UUID, db: Session = Depends(get_db)):
    """Session, active history and accumulated cost in one response."""
    return _detail(db, chat.get_session(db, session_id))


@router.post("/{session_id}/reset", response_model=schemas.SessionDetail)
def reset_session(session_id: uuid.UUID, db: Session = Depends(get_db)):
    """Start a fresh context under the same session id."""
    return _detail(db, chat.reset_session(db, session_id))


@router.get("/{session_id}/messages", response_model=list[schemas.MessageOut])
def list_messages(session_id: uuid.UUID, db: Session = Depends(get_db)):
    """Active history on its own, for clients that do not need the totals."""
    return chat.active_messages(db, chat.get_session(db, session_id))
