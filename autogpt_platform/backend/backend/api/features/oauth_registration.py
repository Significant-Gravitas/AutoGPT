"""Dynamic client registration (RFC 7591), off unless the instance enables it.

MCP clients register themselves when they aren't given a client ID. Anyone can
call this, so it registers public clients only: no secret, PKCE with S256, and
redirect URIs on the loopback interface or over https. The consent page marks
such a client as unreviewed and names where it sends the user.
"""

import logging
from typing import Literal, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from prisma.enums import APIKeyPermission
from pydantic import BaseModel, Field

from backend.api.utils.client_ip import ClientIP
from backend.api.utils.rate_limit import RateLimiter
from backend.data.auth.oauth import register_public_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()
router = APIRouter()

SUPPORTED_GRANT_TYPES = ("authorization_code", "refresh_token")
MAX_REDIRECT_URIS = 5
MAX_REDIRECT_URI_LENGTH = 2000
MAX_CLIENT_NAME_LENGTH = 100
DEFAULT_CLIENT_NAME = "Unnamed application"
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")

# Per caller, then across everyone, so a spread of addresses is bounded too.
_per_ip_limiter = RateLimiter("oauth:register:ip", max_requests=10, window_seconds=3600)
_overall_limiter = RateLimiter("oauth:register", max_requests=500, window_seconds=3600)


class ClientRegistrationRequest(BaseModel):
    """The client metadata a client registers with (RFC 7591 §2).

    Members this server doesn't use, such as `logo_uri`, are ignored.
    """

    redirect_uris: list[str]
    token_endpoint_auth_method: str = "none"
    grant_types: list[str] = list(SUPPORTED_GRANT_TYPES)
    response_types: list[str] = ["code"]
    client_name: Optional[str] = None
    scope: Optional[str] = Field(
        default=None, description="Space-separated; every permission if omitted"
    )


class ClientRegistrationResponse(BaseModel):
    """The client as registered (RFC 7591 §3.2.1). It has no secret."""

    client_id: str
    client_id_issued_at: int
    client_name: str
    redirect_uris: list[str]
    grant_types: list[str]
    response_types: list[str] = ["code"]
    token_endpoint_auth_method: Literal["none"] = "none"
    scope: str


class ClientRegistrationError(BaseModel):
    """RFC 7591 §3.2.2."""

    error: Literal["invalid_redirect_uri", "invalid_client_metadata"]
    error_description: str


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=ClientRegistrationResponse,
    responses={
        400: {"model": ClientRegistrationError},
        404: {"description": "Dynamic client registration is not enabled"},
        429: {"description": "Too many registrations"},
    },
)
async def register_client(
    metadata: ClientRegistrationRequest, ip: ClientIP
) -> ClientRegistrationResponse | JSONResponse:
    """Register a public OAuth client (RFC 7591), as MCP clients do to sign in."""
    if not registration_enabled():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dynamic client registration is not enabled",
        )
    await _per_ip_limiter.check(ip)
    await _overall_limiter.check("all")

    if problem := registration_problem(metadata):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content=problem.model_dump()
        )

    scopes = _requested_scopes(metadata.scope)
    app = await register_public_client(
        name=(metadata.client_name or "").strip() or DEFAULT_CLIENT_NAME,
        redirect_uris=metadata.redirect_uris,
        grant_types=metadata.grant_types,
        scopes=scopes,
    )
    logger.info(f"OAuth client {app.client_id} registered itself")
    return ClientRegistrationResponse(
        client_id=app.client_id,
        client_id_issued_at=int(app.created_at.timestamp()),
        client_name=app.name,
        redirect_uris=app.redirect_uris,
        grant_types=app.grant_types,
        scope=" ".join(s.value for s in scopes),
    )


def registration_enabled() -> bool:
    """The one reader of the setting, so a test that patches it reaches both
    the endpoint and the metadata that advertises it."""
    return settings.config.oauth_dynamic_client_registration


def registration_problem(
    metadata: ClientRegistrationRequest,
) -> Optional[ClientRegistrationError]:
    """Why this metadata can't be registered, or None if it can."""
    if not 1 <= len(metadata.redirect_uris) <= MAX_REDIRECT_URIS:
        return _invalid_redirect(f"Register 1 to {MAX_REDIRECT_URIS} redirect URIs")
    if bad := [uri for uri in metadata.redirect_uris if not _allowed_redirect(uri)]:
        return _invalid_redirect(
            f"Not a loopback http or an https redirect URI: {', '.join(bad)}"
        )
    if metadata.token_endpoint_auth_method != "none":
        return _invalid_metadata(
            "Only public clients can register: token_endpoint_auth_method must be "
            "none"
        )
    if "authorization_code" not in metadata.grant_types or not set(
        metadata.grant_types
    ) <= set(SUPPORTED_GRANT_TYPES):
        return _invalid_metadata(
            "grant_types must include authorization_code and may add refresh_token"
        )
    if metadata.response_types != ["code"]:
        return _invalid_metadata('response_types must be ["code"]')
    if len(metadata.client_name or "") > MAX_CLIENT_NAME_LENGTH:
        return _invalid_metadata(
            f"client_name is longer than {MAX_CLIENT_NAME_LENGTH} characters"
        )
    known = {p.value for p in APIKeyPermission}
    if unknown := [s for s in (metadata.scope or "").split() if s not in known]:
        return _invalid_metadata(f"Unknown scope: {', '.join(unknown)}")
    return None


def _allowed_redirect(uri: str) -> bool:
    """https anywhere, or http on the loopback interface (RFC 8252 §7.3)."""
    if len(uri) > MAX_REDIRECT_URI_LENGTH:
        return False
    parsed = urlparse(uri)
    if parsed.fragment or parsed.username or parsed.password or not parsed.hostname:
        return False
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and parsed.hostname in LOOPBACK_HOSTS


def _requested_scopes(scope: Optional[str]) -> list[APIKeyPermission]:
    if not scope:
        return list(APIKeyPermission)
    return [APIKeyPermission(s) for s in dict.fromkeys(scope.split())]


def _invalid_redirect(description: str) -> ClientRegistrationError:
    return ClientRegistrationError(
        error="invalid_redirect_uri", error_description=description
    )


def _invalid_metadata(description: str) -> ClientRegistrationError:
    return ClientRegistrationError(
        error="invalid_client_metadata", error_description=description
    )
