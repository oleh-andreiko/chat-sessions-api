"""Domain errors and their HTTP representation.

Every error leaves the API in the same shape:
    {"error": {"code": "...", "message": "..."}}
so a client never has to guess which of two formats it got.
"""

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for errors this application raises on purpose."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class SessionNotFound(AppError):
    status_code = 404
    code = "session_not_found"


class UpstreamError(AppError):
    """OpenAI call failed: timeout, rate limit, or a 5xx on their side."""

    status_code = 502
    code = "upstream_error"


class UnsupportedModel(AppError):
    """The caller asked for a model this service has no price for.

    Separate from PricingNotConfigured: this one is the caller's mistake, not a
    broken configuration, so it is a 400 and names the models that do work.
    """

    status_code = 400
    code = "unsupported_model"


class PricingNotConfigured(AppError):
    """A model has no entry in the pricing table — a configuration mistake."""

    status_code = 500
    code = "pricing_not_configured"


def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Reshape FastAPI's own validation errors into the format above.

    Without this, a 422 would be the one response with a different body shape.
    """
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first["loc"])
    return JSONResponse(
        status_code=422,
        content={
            "error": {"code": "invalid_input", "message": f"{field}: {first['msg']}"}
        },
    )


def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """Answer database failures in the same shape as every other error.

    Without this the connection dying mid-request escapes as an unhandled
    exception and the client gets a bare 500 with no body it can parse.
    """
    logger.error("Database error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "database_error", "message": "Database unavailable."}},
    )


# Framework-raised errors carry no domain meaning, so they get a code derived
# from the status. Anything unlisted falls back to a generic one.
_HTTP_ERROR_CODES = {
    400: "invalid_request",
    404: "not_found",
    405: "method_not_allowed",
}


def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Reshape errors raised by FastAPI itself into the one error format.

    An unreadable request body and an unknown route are raised by the framework
    as its own HTTPException, which would otherwise answer with {"detail": ...}
    and make these two the only replies in the API speaking a different shape.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": _HTTP_ERROR_CODES.get(exc.status_code, "http_error"),
                "message": str(exc.detail),
            }
        },
    )
