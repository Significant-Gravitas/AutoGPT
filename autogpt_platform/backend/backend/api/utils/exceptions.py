"""
Shared exception handlers for FastAPI applications.

Provides a single `add_exception_handlers` function that registers a consistent
set of exception-to-HTTP-status mappings on any FastAPI app instance. This
ensures that all mounted sub-apps (v1, v2, main) handle errors uniformly.
"""

import json
import logging
import reprlib
from typing import Any, Sequence

import fastapi
import fastapi.responses
import pydantic
from fastapi.exceptions import RequestValidationError
from prisma.errors import PrismaError
from prisma.errors import RecordNotFoundError as PrismaRecordNotFoundError
from starlette import status

from backend.api.features.library.exceptions import (
    FolderAlreadyExistsError,
    FolderValidationError,
)
from backend.copilot.rate_limit import UserPaywalledError
from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError
from backend.util.exceptions import (
    ConflictError,
    MissingConfigError,
    NotAuthorizedError,
    NotFoundError,
    PreconditionFailed,
)
from backend.util.url_redaction import redact_url

logger = logging.getLogger(__name__)


def add_exception_handlers(
    app: fastapi.FastAPI, *, server_error_detail: bool = True
) -> None:
    """
    Register standard exception handlers on the given FastAPI app.

    Mounted sub-apps do NOT inherit exception handlers from the parent app,
    so each app instance must register its own handlers.

    ``server_error_detail=False`` keeps the exception text out of 5xx bodies,
    for an app third parties call: it can name internal services and queries.
    """

    def _handle_error(status_code: int = 500, log_error: bool = True):
        return _error_handler(status_code, log_error, server_error_detail)

    for exception, handler in {
        # It's the client's problem: HTTP 4XX
        NotFoundError: _handle_error(status.HTTP_404_NOT_FOUND),
        NotAuthorizedError: _handle_error(status.HTTP_403_FORBIDDEN),
        # UserPaywalledError raised at deep enqueue paths (e.g. add_graph_execution)
        # maps to HTTP 402. log_error=False because it's not an error from the
        # server's perspective — the user just lacks entitlement.
        UserPaywalledError: _handle_error(
            status.HTTP_402_PAYMENT_REQUIRED, log_error=False
        ),
        PreconditionFailed: _handle_error(status.HTTP_428_PRECONDITION_REQUIRED),
        RequestValidationError: _handle_validation_error,
        pydantic.ValidationError: _handle_validation_error,
        PrismaRecordNotFoundError: _handle_error(status.HTTP_404_NOT_FOUND),
        FolderAlreadyExistsError: _handle_error(status.HTTP_409_CONFLICT),
        ConflictError: _handle_error(status.HTTP_409_CONFLICT),
        FolderValidationError: _handle_error(status.HTTP_400_BAD_REQUEST),
        # Bad/revoked node credentials during graph activation are the
        # client's problem, not an opaque 500.
        GraphActivationError: _handle_error(status.HTTP_400_BAD_REQUEST),
        ValueError: _handle_error(status.HTTP_400_BAD_REQUEST),
        # It's the backend's problem: HTTP 5XX
        MissingConfigError: _handle_error(status.HTTP_503_SERVICE_UNAVAILABLE),
        PrismaError: _handle_error(status.HTTP_500_INTERNAL_SERVER_ERROR),
        Exception: _handle_error(status.HTTP_500_INTERNAL_SERVER_ERROR),
    }.items():
        app.add_exception_handler(exception, handler)


def _error_handler(status_code: int, log_error: bool, server_error_detail: bool):
    def handler(request: fastapi.Request, exc: Exception):
        if log_error:
            if status_code >= 500:
                logger.exception(
                    "%s %s failed. Investigate and resolve the underlying issue: %s",
                    request.method,
                    redact_url(request.url.path),
                    exc,
                    exc_info=exc,
                )
            else:
                logger.warning(
                    "%s %s failed with %d: %s",
                    request.method,
                    redact_url(request.url.path),
                    status_code,
                    exc,
                )

        hint = (
            "Adjust the request and retry."
            if status_code < 500
            else "Check server logs and dependent services."
        )
        return fastapi.responses.JSONResponse(
            content={
                "message": f"Failed to process {request.method} {request.url.path}",
                "detail": (
                    str(exc)
                    if status_code < 500 or server_error_detail
                    else "Internal server error"
                ),
                "hint": hint,
            },
            status_code=status_code,
        )

    return handler


# A validation error response and its log line name at most this many errors.
MAX_VALIDATION_ERRORS_SHOWN = 50
# Each named error shows the value it failed on as a preview of at most this
# many characters.
MAX_VALIDATION_INPUT_PREVIEW_CHARS = 200

_input_preview = reprlib.Repr()
_input_preview.maxlevel = 3
_input_preview.maxstring = MAX_VALIDATION_INPUT_PREVIEW_CHARS
_input_preview.maxother = MAX_VALIDATION_INPUT_PREVIEW_CHARS


async def _handle_validation_error(
    request: fastapi.Request, exc: Exception
) -> fastapi.responses.Response:
    errors: Sequence | str
    omitted = 0
    if isinstance(exc, (RequestValidationError, pydantic.ValidationError)):
        all_errors = exc.errors()
        errors = [
            _bounded_error(error)
            for error in all_errors[:MAX_VALIDATION_ERRORS_SHOWN]
        ]
        omitted = len(all_errors) - len(errors)
    else:
        errors = str(exc)
    logger.error(
        "Validation failed for %s %s: %s%s. Fix the request payload and try again.",
        request.method,
        redact_url(request.url.path),
        errors,
        f" (and {omitted} more errors)" if omitted else "",
    )

    response_content: dict[str, Any] = {
        "message": f"Invalid data for {request.method} {request.url.path}",
        "detail": errors,
        "hint": "Ensure the request matches the API schema.",
    }
    if omitted:
        response_content["omitted_errors"] = omitted

    content_json = json.dumps(response_content, default=str)

    return fastapi.responses.Response(
        content=content_json,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        media_type="application/json",
    )


def _bounded_error(error: Any) -> dict[str, Any]:
    """An error's type, location and message, and a short preview of the
    value it failed on."""
    bounded = {key: error[key] for key in ("type", "loc", "msg") if key in error}
    if "input" in error:
        bounded["input"] = _preview(error["input"])
    return bounded


def _preview(value: Any) -> str:
    try:
        text = _input_preview.repr(value)
    except ValueError:
        text = f"<{type(value).__name__}>"
    return text[:MAX_VALIDATION_INPUT_PREVIEW_CHARS]
