"""
V2 External API - Error responses

One body shape for every non-2xx: `{"error": {"code", "message", "details"}}`.

The shared handlers in `api/utils/exceptions` emit `{"message", "detail",
"hint"}` and FastAPI's own `HTTPException` path emits `{"detail": ...}`, so v2
registers its own handlers over both rather than inheriting either.
"""

import logging
from typing import Optional

import fastapi
import fastapi.responses
import pydantic
from fastapi.exceptions import RequestValidationError
from prisma.errors import PrismaError
from prisma.errors import RecordNotFoundError as PrismaRecordNotFoundError
from pydantic import BaseModel, Field, JsonValue
from starlette import status
from starlette.exceptions import HTTPException as StarletteHTTPException

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

logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable identifier for the failure")
    message: str = Field(description="Human-readable explanation")
    details: Optional[dict[str, JsonValue]] = Field(
        default=None, description="Structured context, when the failure has any"
    )


class ErrorResponse(BaseModel):
    """The body of every v2 response that is not 2xx."""

    error: ErrorDetail


def add_v2_exception_handlers(app: fastapi.FastAPI) -> None:
    """Register the v2 error envelope over every exception that can reach a route."""
    for exception, status_code in {
        NotFoundError: status.HTTP_404_NOT_FOUND,
        PrismaRecordNotFoundError: status.HTTP_404_NOT_FOUND,
        NotAuthorizedError: status.HTTP_403_FORBIDDEN,
        # Not an error from the server's perspective — the user lacks entitlement.
        UserPaywalledError: status.HTTP_402_PAYMENT_REQUIRED,
        PreconditionFailed: status.HTTP_428_PRECONDITION_REQUIRED,
        FolderAlreadyExistsError: status.HTTP_409_CONFLICT,
        ConflictError: status.HTTP_409_CONFLICT,
        FolderValidationError: status.HTTP_400_BAD_REQUEST,
        GraphActivationError: status.HTTP_400_BAD_REQUEST,
        ValueError: status.HTTP_400_BAD_REQUEST,
        MissingConfigError: status.HTTP_503_SERVICE_UNAVAILABLE,
        PrismaError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        Exception: status.HTTP_500_INTERNAL_SERVER_ERROR,
    }.items():
        app.add_exception_handler(exception, _handler_for(status_code))

    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(pydantic.ValidationError, _handle_validation_error)


def error_response(
    status_code: int,
    message: str,
    code: Optional[str] = None,
    details: Optional[dict[str, JsonValue]] = None,
) -> fastapi.responses.JSONResponse:
    """Build a v2 error body. Also used outside the handlers, by the middleware."""
    body = ErrorResponse(
        error=ErrorDetail(
            code=code or error_code_for(status_code),
            message=message,
            details=details,
        )
    )
    return fastapi.responses.JSONResponse(
        content=body.model_dump(mode="json"), status_code=status_code
    )


def error_code_for(status_code: int) -> str:
    default = "internal_error" if status_code >= 500 else "request_failed"
    return _ERROR_CODES.get(status_code, default)


_ERROR_CODES = {
    status.HTTP_400_BAD_REQUEST: "bad_request",
    status.HTTP_401_UNAUTHORIZED: "unauthorized",
    status.HTTP_402_PAYMENT_REQUIRED: "payment_required",
    status.HTTP_403_FORBIDDEN: "forbidden",
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
    status.HTTP_409_CONFLICT: "conflict",
    status.HTTP_413_CONTENT_TOO_LARGE: "payload_too_large",
    status.HTTP_422_UNPROCESSABLE_CONTENT: "validation_error",
    status.HTTP_428_PRECONDITION_REQUIRED: "precondition_required",
    status.HTTP_429_TOO_MANY_REQUESTS: "rate_limit_exceeded",
    status.HTTP_500_INTERNAL_SERVER_ERROR: "internal_error",
    status.HTTP_502_BAD_GATEWAY: "upstream_error",
    status.HTTP_503_SERVICE_UNAVAILABLE: "service_unavailable",
}


def _handler_for(status_code: int):
    def handler(
        request: fastapi.Request, exc: Exception
    ) -> fastapi.responses.JSONResponse:
        _log(request, status_code, exc)
        # A 5xx message describes our internals, not the caller's request.
        message = (
            f"Failed to process {request.method} {request.url.path}"
            if status_code >= 500
            else (str(exc) or type(exc).__name__)
        )
        return error_response(status_code, message)

    return handler


def _handle_http_exception(
    request: fastapi.Request, exc: Exception
) -> fastapi.responses.JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    _log(request, exc.status_code, exc)
    # `detail` is a dict when a route raised HTTPException with structured context.
    if isinstance(exc.detail, dict):
        message = str(exc.detail.get("message", "Request failed"))
        details = {k: v for k, v in exc.detail.items() if k != "message"}
    else:
        message = str(exc.detail)
        details = None
    return error_response(exc.status_code, message, details=details)


def _handle_validation_error(
    request: fastapi.Request, exc: Exception
) -> fastapi.responses.JSONResponse:
    _log(request, status.HTTP_422_UNPROCESSABLE_CONTENT, exc)
    errors = (
        exc.errors()
        if isinstance(exc, (RequestValidationError, pydantic.ValidationError))
        else []
    )
    return error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        f"Invalid data for {request.method} {request.url.path}",
        details={
            "errors": pydantic.TypeAdapter(JsonValue).dump_python(
                errors, mode="json", warnings=False
            )
        },
    )


def _log(request: fastapi.Request, status_code: int, exc: Exception) -> None:
    if status_code >= 500:
        logger.exception(
            f"{request.method} {request.url.path} failed: {exc}", exc_info=exc
        )
    elif status_code not in (
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_403_FORBIDDEN,
        status.HTTP_404_NOT_FOUND,
    ):
        logger.warning(f"{request.method} {request.url.path} -> {status_code}: {exc}")
