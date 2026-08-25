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
    answer = chat.send_message(db, session_id, payload.content)
    return schemas.MessageReply(
        message=schemas.MessageOut.model_validate(answer),
        model=answer.model,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        cost=answer.cost,
    )


@router.get("/{session_id}", response_model=schemas.SessionDetail)
def get_session(session_id: uuid.UUID, db: Session = Depends(get_db)):
    """Session, full history and accumulated cost in one response."""
    session = chat.get_session(db, session_id)

    # Totals are summed on read instead of being kept as counters on the
    # session row: no denormalised value can drift out of sync this way, and at
    # this data volume the sum costs nothing.
    return schemas.SessionDetail(
        id=session.id,
        title=session.title,
        model=session.model,
        created_at=session.created_at,
        messages=[schemas.MessageOut.model_validate(m) for m in session.messages],
        total_prompt_tokens=sum(m.prompt_tokens or 0 for m in session.messages),
        total_completion_tokens=sum(m.completion_tokens or 0 for m in session.messages),
        total_cost=sum((m.cost or Decimal(0) for m in session.messages), Decimal(0)),
    )


@router.get("/{session_id}/messages", response_model=list[schemas.MessageOut])
def list_messages(session_id: uuid.UUID, db: Session = Depends(get_db)):
    """History on its own, for clients that do not need the totals."""
    return chat.get_session(db, session_id).messages
