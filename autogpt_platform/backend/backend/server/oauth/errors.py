"""
OAuth 2.0 Error Responses (RFC 6749 Section 5.2).
"""

from enum import Enum
from typing import Optional
from urllib.parse import urlencode

from fastapi import HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel


class OAuthErrorCode(str, Enum):
    """Standard OAuth 2.0 error codes."""

    # Authorization endpoint errors (RFC 6749 Section 4.1.2.1)
    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED_CLIENT = "unauthorized_client"
    ACCESS_DENIED = "access_denied"
    UNSUPPORTED_RESPONSE_TYPE = "unsupported_response_type"
    INVALID_SCOPE = "invalid_scope"
    SERVER_ERROR = "server_error"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"

    # Token endpoint errors (RFC 6749 Section 5.2)
    INVALID_CLIENT = "invalid_client"
    INVALID_GRANT = "invalid_grant"
    UNSUPPORTED_GRANT_TYPE = "unsupported_grant_type"

    # Extension errors
    LOGIN_REQUIRED = "login_required"
    CONSENT_REQUIRED = "consent_required"


class OAuthErrorResponse(BaseModel):
    """OAuth error response model."""

    error: str
    error_description: Optional[str] = None
    error_uri: Optional[str] = None


class OAuthError(Exception):
    """Base OAuth error exception."""

    def __init__(
        self,
        error: OAuthErrorCode,
        description: Optional[str] = None,
        uri: Optional[str] = None,
        state: Optional[str] = None,
    ):
        self.error = error
        self.description = description
        self.uri = uri
        self.state = state
        super().__init__(description or error.value)

    def to_response(self) -> OAuthErrorResponse:
        """Convert to response model."""
        return OAuthErrorResponse(
            error=self.error.value,
            error_description=self.description,
            error_uri=self.uri,
        )

    def to_redirect(self, redirect_uri: str) -> RedirectResponse:
        """Convert to redirect response with error in query params."""
        params = {"error": self.error.value}
        if self.description:
            params["error_description"] = self.description
        if self.uri:
            params["error_uri"] = self.uri
        if self.state:
            params["state"] = self.state

        separator = "&" if "?" in redirect_uri else "?"
        url = f"{redirect_uri}{separator}{urlencode(params)}"
        return RedirectResponse(url=url, status_code=302)

    def to_http_exception(self, status_code: int = 400) -> HTTPException:
        """Convert to FastAPI HTTPException."""
        return HTTPException(
            status_code=status_code,
            detail=self.to_response().model_dump(exclude_none=True),
        )


# Convenience error classes
class InvalidRequestError(OAuthError):
    """The request is missing a required parameter or is otherwise malformed."""

    def __init__(self, description: str, state: Optional[str] = None):
        super().__init__(OAuthErrorCode.INVALID_REQUEST, description, state=state)


class UnauthorizedClientError(OAuthError):
    """The client is not authorized to request an authorization code."""

    def __init__(self, description: str, state: Optional[str] = None):
        super().__init__(OAuthErrorCode.UNAUTHORIZED_CLIENT, description, state=state)


class AccessDeniedError(OAuthError):
    """The resource owner denied the request."""

    def __init__(self, description: str = "Access denied", state: Optional[str] = None):
        super().__init__(OAuthErrorCode.ACCESS_DENIED, description, state=state)


class InvalidScopeError(OAuthError):
    """The requested scope is invalid, unknown, or malformed."""

    def __init__(self, description: str, state: Optional[str] = None):
        super().__init__(OAuthErrorCode.INVALID_SCOPE, description, state=state)


class InvalidClientError(OAuthError):
    """Client authentication failed."""

    def __init__(self, description: str = "Invalid client"):
        super().__init__(OAuthErrorCode.INVALID_CLIENT, description)


class InvalidGrantError(OAuthError):
    """The provided authorization code or refresh token is invalid."""

    def __init__(self, description: str = "Invalid grant"):
        super().__init__(OAuthErrorCode.INVALID_GRANT, description)


class UnsupportedGrantTypeError(OAuthError):
    """The authorization grant type is not supported."""

    def __init__(self, grant_type: str):
        super().__init__(
            OAuthErrorCode.UNSUPPORTED_GRANT_TYPE,
            f"Grant type '{grant_type}' is not supported",
        )


class LoginRequiredError(OAuthError):
    """User must be logged in to complete the request."""

    def __init__(self, state: Optional[str] = None):
        super().__init__(
            OAuthErrorCode.LOGIN_REQUIRED,
            "User authentication required",
            state=state,
        )


class ConsentRequiredError(OAuthError):
    """User consent is required for the requested scopes."""

    def __init__(self, state: Optional[str] = None):
        super().__init__(
            OAuthErrorCode.CONSENT_REQUIRED,
            "User consent required",
            state=state,
        )
