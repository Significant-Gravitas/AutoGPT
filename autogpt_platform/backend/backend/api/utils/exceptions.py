"""
Shared exception handlers for FastAPI applications.

Provides a single `add_exception_handlers` function that registers a consistent
set of exception-to-HTTP-status mappings on any FastAPI app instance. This
ensures that all mounted sub-apps (v1, v2, main) handle errors uniformly.
"""

import json
import logging

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
from backend.util.exceptions import (
    MissingConfigError,
    NotAuthorizedError,
    NotFoundError,
    PreconditionFailed,
)

logger = logging.getLogger(__name__)


def add_exception_handlers(app: fastapi.FastAPI) -> None:
    """
    Register standard exception handlers on the given FastAPI app.

    Mounted sub-apps do NOT inherit exception handlers from the parent app,
    so each app instance must register its own handlers.
    """
    for exception, handler in {
        # It's the client's problem: HTTP 4XX
        NotFoundError: _handle_error(status.HTTP_404_NOT_FOUND, log_error=False),
        NotAuthorizedError: _handle_error(status.HTTP_403_FORBIDDEN, log_error=False),
        PreconditionFailed: _handle_error(status.HTTP_428_PRECONDITION_REQUIRED),
        RequestValidationError: _handle_validation_error,
        pydantic.ValidationError: _handle_validation_error,
        PrismaRecordNotFoundError: _handle_error(status.HTTP_404_NOT_FOUND),
        FolderAlreadyExistsError: _handle_error(
            status.HTTP_409_CONFLICT, log_error=False
        ),
        FolderValidationError: _handle_error(
            status.HTTP_400_BAD_REQUEST, log_error=False
        ),
        ValueError: _handle_error(status.HTTP_400_BAD_REQUEST),
        # It's the backend's problem: HTTP 5XX
        MissingConfigError: _handle_error(status.HTTP_503_SERVICE_UNAVAILABLE),
        PrismaError: _handle_error(status.HTTP_500_INTERNAL_SERVER_ERROR),
        Exception: _handle_error(status.HTTP_500_INTERNAL_SERVER_ERROR),
    }.items():
        app.add_exception_handler(exception, handler)


def _handle_error(status_code: int = 500, log_error: bool = True):
    def handler(request: fastapi.Request, exc: Exception):
        if log_error:
            logger.exception(
                "%s %s failed. Investigate and resolve the underlying issue: %s",
                request.method,
                request.url.path,
                exc,
                exc_info=exc,
            )

        hint = (
            "Adjust the request and retry."
            if status_code < 500
            else "Check server logs and dependent services."
        )
        return fastapi.responses.JSONResponse(
            content={
                "message": f"Failed to process {request.method} {request.url.path}",
                "detail": str(exc),
                "hint": hint,
            },
            status_code=status_code,
        )

    return handler


async def _handle_validation_error(
    request: fastapi.Request, exc: Exception
) -> fastapi.responses.Response:
    logger.error(
        "Validation failed for %s %s: %s. Fix the request payload and try again.",
        request.method,
        request.url.path,
        exc,
    )
    errors: list | str
    if hasattr(exc, "errors"):
        errors = exc.errors()  # type: ignore[call-arg]
    else:
        errors = str(exc)

    response_content = {
        "message": f"Invalid data for {request.method} {request.url.path}",
        "detail": errors,
        "hint": "Ensure the request matches the API schema.",
    }

    content_json = json.dumps(response_content)

    return fastapi.responses.Response(
        content=content_json,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        media_type="application/json",
    )
