"""Application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse

from app.db import engine
from app.errors import AppError, app_error_handler, validation_error_handler
from app.models import Base
from app.routers import sessions

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Two tables and no production history to migrate, so the schema is created
    # on startup. A migration tool would be the answer for a real deployment.
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="Chat sessions API", lifespan=lifespan)
app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.include_router(sessions.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ui", include_in_schema=False)
def ui():
    """A one-file page for clicking through the API by hand.

    Served directly instead of mounting a static directory: there is exactly
    one file, and it needs no build step or extra dependency.
    """
    return FileResponse(Path(__file__).parent / "static" / "index.html")
