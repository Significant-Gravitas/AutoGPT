"""
OIDC Discovery endpoints.

Implements:
- GET /.well-known/openid-configuration - OIDC Discovery Document
- GET /.well-known/jwks.json - JSON Web Key Set
"""

from fastapi import APIRouter

from backend.server.oauth.models import JWKS, OpenIDConfiguration
from backend.server.oauth.token_service import get_token_service
from backend.util.settings import Settings

discovery_router = APIRouter(tags=["oidc-discovery"])


@discovery_router.get(
    "/.well-known/openid-configuration",
    response_model=OpenIDConfiguration,
)
async def openid_configuration() -> OpenIDConfiguration:
    """
    OIDC Discovery Document.

    Returns metadata about the OAuth 2.0 authorization server including
    endpoints, supported features, and algorithms.
    """
    settings = Settings()
    base_url = settings.config.platform_base_url or "https://platform.agpt.co"

    return OpenIDConfiguration(
        issuer=base_url,
        authorization_endpoint=f"{base_url}/oauth/authorize",
        token_endpoint=f"{base_url}/oauth/token",
        userinfo_endpoint=f"{base_url}/oauth/userinfo",
        revocation_endpoint=f"{base_url}/oauth/revoke",
        jwks_uri=f"{base_url}/.well-known/jwks.json",
        scopes_supported=[
            "openid",
            "profile",
            "email",
            "integrations:list",
            "integrations:connect",
            "integrations:delete",
            "agents:execute",
        ],
        response_types_supported=["code"],
        grant_types_supported=["authorization_code", "refresh_token"],
        token_endpoint_auth_methods_supported=[
            "client_secret_post",
            "client_secret_basic",
            "none",  # For public clients with PKCE
        ],
        code_challenge_methods_supported=["S256"],
        subject_types_supported=["public"],
        id_token_signing_alg_values_supported=["RS256"],
    )


@discovery_router.get("/.well-known/jwks.json", response_model=JWKS)
async def jwks() -> dict:
    """
    JSON Web Key Set (JWKS).

    Returns the public key(s) used to verify JWT signatures.
    External applications can use these keys to verify access tokens
    and ID tokens issued by this authorization server.
    """
    token_service = get_token_service()
    return token_service.get_jwks()
