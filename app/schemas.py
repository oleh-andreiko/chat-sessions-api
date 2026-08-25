"""Request and response shapes. Validation happens here, before any service runs."""

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

# Pydantic reserves the "model_" prefix for its own attributes and warns about
# fields that look like it. Our field really is called "model", so the guard is
# switched off for these schemas.
_ALLOW_MODEL_FIELD = ConfigDict(protected_namespaces=(), from_attributes=True)


class SessionCreate(BaseModel):
    model_config = _ALLOW_MODEL_FIELD

    model: str | None = None
    title: str | None = None


class SessionOut(BaseModel):
    model_config = _ALLOW_MODEL_FIELD

    id: uuid.UUID
    title: str | None
    model: str
    created_at: datetime


class MessageCreate(BaseModel):
    # min_length=1 rejects an empty prompt before it costs an OpenAI call.
    content: str = Field(min_length=1)


class MessageOut(BaseModel):
    model_config = _ALLOW_MODEL_FIELD

    id: int
    role: str
    content: str
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost: Decimal | None
    created_at: datetime


class MessageReply(BaseModel):
    """What POST /sessions/{id}/messages returns: the reply plus what it cost."""

    model_config = _ALLOW_MODEL_FIELD

    message: MessageOut
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: Decimal


class SessionDetail(SessionOut):
    """Session, full history and accumulated cost in one response."""

    messages: list[MessageOut]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: Decimal
