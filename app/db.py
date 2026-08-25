"""Database engine and the per-request session dependency."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import config

engine = create_engine(
    config.DATABASE_URL,
    # Check a pooled connection before handing it out: after the database
    # restarts, the pool still holds sockets that look open but are not.
    pool_pre_ping=True,
    # Without a bound, connecting to a database that is down hangs the request
    # instead of failing.
    connect_args={"connect_timeout": 5},
)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    """FastAPI dependency: one database session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
