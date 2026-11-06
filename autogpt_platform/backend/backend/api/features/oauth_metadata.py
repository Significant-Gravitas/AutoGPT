"""RFC 8414 discovery for the platform's OAuth provider.

An MCP client finds the provider through the MCP server's resource metadata,
which names `oauth_issuer()`; this document then tells it where to send the
user to consent and where to redeem the code.
"""

from mcp.shared.auth import OAuthMetadata
from prisma.enums import APIKeyPermission
from pydantic import AnyHttpUrl

from backend.util.settings import Settings

# RFC 8414 §3.1: an issuer with no path has its document at the origin root.
WELL_KNOWN_AUTHORIZATION_SERVER_PATH = "/.well-known/oauth-authorization-server"

# The consent page, which posts the user's approval to `/authorize` on this API.
AUTHORIZE_PAGE_PATH = "/auth/authorize"


def oauth_issuer() -> AnyHttpUrl:
    """The provider's issuer: the backend origin, where its metadata is served."""
    return AnyHttpUrl(Settings().config.platform_base_url or "https://platform.agpt.co")


def authorization_server_metadata(token_path: str) -> OAuthMetadata:
    """What the provider does, for a client that has only its issuer.

    `token_path` is where the token route is mounted, read from the app so the
    document follows the route if it moves.
    """
    issuer = oauth_issuer()
    frontend = Settings().config.frontend_base_url or "http://localhost:3000"
    return OAuthMetadata(
        issuer=issuer,
        authorization_endpoint=AnyHttpUrl(f"{frontend}{AUTHORIZE_PAGE_PATH}"),
        token_endpoint=AnyHttpUrl(f"{str(issuer).rstrip('/')}{token_path}"),
        scopes_supported=[p.value for p in APIKeyPermission],
        response_types_supported=["code"],
        grant_types_supported=["authorization_code", "refresh_token"],
        token_endpoint_auth_methods_supported=[
            "client_secret_basic",
            "client_secret_post",
        ],
        # `/authorize` refuses a request without a code challenge.
        code_challenge_methods_supported=["S256", "plain"],
    )
