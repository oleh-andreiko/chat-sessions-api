"""SQLAlchemy models: a chat session and the messages inside it."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ChatSession(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=True)
    # The model is fixed when the session is created, so every reply in one
    # session is priced with the same rates.
    model = Column(Text, nullable=False)
    # Reset does not delete anything: it moves the session to the next
    # generation, and only messages of the current one count as active history.
    current_generation = Column(Integer, nullable=False, default=1, server_default="1")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at, Message.id",
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Which generation of the session this belongs to. Rows from earlier
    # generations stay in the table as the archive of what was already paid for.
    generation = Column(Integer, nullable=False, default=1, server_default="1")
    role = Column(Text, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)

    # The four columns below are filled for assistant messages only: a user
    # message costs nothing on its own and has no usage of its own.
    # model is the alias the cost was priced by; resolved_model is what OpenAI
    # reported serving the request. They differ: aliases resolve to snapshots.
    model = Column(Text, nullable=True)
    resolved_model = Column(Text, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    # Numeric, not float: money must not accumulate binary rounding error.
    # Cost is stored as calculated at answer time, because OpenAI prices change
    # and recomputing old sessions later would rewrite their history.
    #
    # Scale 8, not 6: at gpt-4o-mini rates a short exchange costs around
    # $0.000007, so six decimal places would round away a noticeable part of it.
    cost = Column(Numeric(14, 8), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    session = relationship("ChatSession", back_populates="messages")

    __table_args__ = (
        Index(
            "ix_messages_session_generation_created",
            "session_id",
            "generation",
            "created_at",
        ),
    )
