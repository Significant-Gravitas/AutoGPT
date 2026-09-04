"""
MCP (Model Context Protocol) API routes.

Provides endpoints for MCP tool discovery and OAuth authentication so the
frontend can list available tools on an MCP server before placing a block.
"""

import logging
from typing import Annotated, Any

import fastapi
from autogpt_libs.auth import get_user_id
from fastapi import Security
from pydantic import BaseModel, Field, SecretStr

from backend.api.features.integrations.router import (
    CredentialsMetaResponse,
    to_meta_response,
)
from backend.blocks.mcp.client import (
    MCPClient,
    MCPClientError,
    normalize_mcp_authorization,
)
from backend.blocks.mcp.helpers import (
    auto_lookup_mcp_credential,
    is_manual_mcp_credential,
    mcp_authorization_header,
    normalize_mcp_url,
    server_host,
)
from backend.blocks.mcp.oauth import MCPOAuthHandler
from backend.data.model import OAuth2Credentials
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.providers import ProviderName
from backend.util.request import HTTPClientError, Requests, validate_url_host
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()
router = fastapi.APIRouter(tags=["mcp"])
creds_manager = IntegrationCredentialsManager()


# ====================== Tool Discovery ====================== #


# Bounded so the per-character control-character scan in
# `normalize_mcp_authorization` never runs over an unbounded body.  Generous
# enough for a Base64 `user:password` or a long JWT.
_MAX_CREDENTIAL_LENGTH = 8192


class DiscoverToolsRequest(BaseModel):
    """Request to discover tools on an MCP server."""

    server_url: str = Field(description="URL of the MCP server")
    auth_token: SecretStr | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_CREDENTIAL_LENGTH,
        description=(
            "Optional bare Bearer token, Basic/Bearer value, or complete "
            "Authorization header. Omit the field (or send null) for an "
            "unauthenticated server; an empty string is rejected."
        ),
    )


class MCPToolResponse(BaseModel):
    """A single MCP tool returned by discovery."""

    name: str
    description: str
    input_schema: dict[str, Any]


class DiscoverToolsResponse(BaseModel):
    """Response containing the list of tools available on an MCP server."""

    tools: list[MCPToolResponse]
    server_name: str | None = None
    protocol_version: str | None = None


@router.post(
    "/discover-tools",
    summary="Discover available tools on an MCP server",
    response_model=DiscoverToolsResponse,
)
async def discover_tools(
    request: DiscoverToolsRequest,
    user_id: Annotated[str, Security(get_user_id)],
) -> DiscoverToolsResponse:
    """
    Connect to an MCP server and return its available tools.

    If the user has a stored MCP credential for this server URL, it will be
    used automatically — no need to pass an explicit auth credential.
    """
    # Validate URL to prevent SSRF — blocks loopback and private IP ranges.
    try:
        await validate_url_host(request.server_url)
    except ValueError as e:
        raise fastapi.HTTPException(status_code=400, detail=f"Invalid server URL: {e}")

    authorization: str | None = None
    explicit_token = (
        request.auth_token.get_secret_value() if request.auth_token else None
    )
    if explicit_token:
        # Fresh user input: normalize exactly here, once.
        try:
            authorization = normalize_mcp_authorization(explicit_token)
        except ValueError as e:
            raise fastapi.HTTPException(status_code=422, detail=str(e)) from e
    else:
        # Auto-use stored MCP credential when no explicit token is provided.
        stored_credential = await auto_lookup_mcp_credential(
            user_id, normalize_mcp_url(request.server_url)
        )
        if stored_credential:
            authorization = mcp_authorization_header(stored_credential)

    client = MCPClient(request.server_url, authorization=authorization)

    try:
        init_result = await client.initialize()
        tools = await client.list_tools()
    except HTTPClientError as e:
        if e.status_code in (401, 403):
            raise fastapi.HTTPException(
                status_code=401,
                detail="This MCP server requires authentication. "
                "Please provide a valid auth credential.",
            )
        raise fastapi.HTTPException(status_code=502, detail=str(e))
    except MCPClientError as e:
        raise fastapi.HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=502,
            detail=f"Failed to connect to MCP server: {e}",
        )
    finally:
        # Discovery now runs on every connect surface, so without this each
        # attempt leaves a session row on the remote until its timeout sweep.
        await client.close()

    return DiscoverToolsResponse(
        tools=[
            MCPToolResponse(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
            )
            for t in tools
        ],
        server_name=(
            init_result.get("serverInfo", {}).get("name")
            or server_host(request.server_url)
            or "MCP"
        ),
        protocol_version=init_result.get("protocolVersion"),
    )


# ======================== OAuth Flow ======================== #


class MCPOAuthLoginRequest(BaseModel):
    """Request to start an OAuth flow for an MCP server."""

    server_url: str = Field(description="URL of the MCP server that requires OAuth")


class MCPOAuthLoginResponse(BaseModel):
    """Response with the OAuth login URL for the user to authenticate."""

    login_url: str
    state_token: str


@router.post(
    "/oauth/login",
    summary="Initiate OAuth login for an MCP server",
)
async def mcp_oauth_login(
    request: MCPOAuthLoginRequest,
    user_id: Annotated[str, Security(get_user_id)],
) -> MCPOAuthLoginResponse:
    """
    Discover OAuth metadata from the MCP server and return a login URL.

    1. Discovers the protected-resource metadata (RFC 9728)
    2. Fetches the authorization server metadata (RFC 8414)
    3. Performs Dynamic Client Registration (RFC 7591) if available
    4. Returns the authorization URL for the frontend to open in a popup
    """
    # Validate URL to prevent SSRF — blocks loopback and private IP ranges.
    try:
        await validate_url_host(request.server_url)
    except ValueError as e:
        raise fastapi.HTTPException(status_code=400, detail=f"Invalid server URL: {e}")

    # Normalize the URL so that credentials stored here are matched consistently
    # by auto_lookup_mcp_credential (which also uses normalized URLs).
    server_url = normalize_mcp_url(request.server_url)
    client = MCPClient(server_url)

    # Step 1: Discover protected-resource metadata (RFC 9728)
    protected_resource = await client.discover_auth()

    metadata: dict[str, Any] | None = None

    if protected_resource and protected_resource.get("authorization_servers"):
        auth_server_url = protected_resource["authorization_servers"][0]
        resource_url = protected_resource.get("resource", server_url)

        # Validate the auth server URL from metadata to prevent SSRF.
        try:
            await validate_url_host(auth_server_url)
        except ValueError as e:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"Invalid authorization server URL in metadata: {e}",
            )

        # Step 2a: Discover auth-server metadata (RFC 8414)
        metadata = await client.discover_auth_server_metadata(auth_server_url)
    else:
        # Fallback: Some MCP servers (e.g. Linear) are their own auth server
        # and serve OAuth metadata directly without protected-resource metadata.
        # Don't assume a resource_url — omitting it lets the auth server choose
        # the correct audience for the token (RFC 8707 resource is optional).
        resource_url = None
        metadata = await client.discover_auth_server_metadata(server_url)

    if (
        not metadata
        or "authorization_endpoint" not in metadata
        or "token_endpoint" not in metadata
    ):
        raise fastapi.HTTPException(
            status_code=400,
            detail="This MCP server does not advertise OAuth support. "
            "You may need to provide an auth credential manually.",
        )

    authorize_url = metadata["authorization_endpoint"]
    token_url = metadata["token_endpoint"]
    registration_endpoint = metadata.get("registration_endpoint")
    revoke_url = metadata.get("revocation_endpoint")

    # Step 3: Dynamic Client Registration (RFC 7591) if available
    frontend_base_url = settings.config.frontend_base_url
    if not frontend_base_url:
        raise fastapi.HTTPException(
            status_code=500,
            detail="Frontend base URL is not configured.",
        )
    redirect_uri = f"{frontend_base_url}/auth/integrations/mcp_callback"

    client_id = ""
    client_secret = ""
    if registration_endpoint:
        # Validate the registration endpoint from metadata to prevent SSRF.
        try:
            await validate_url_host(registration_endpoint)
        except ValueError:
            pass  # Skip registration, fall back to default client_id
        else:
            reg_result = await _register_mcp_client(
                registration_endpoint, redirect_uri, server_url
            )
            if reg_result:
                client_id = reg_result.get("client_id", "")
                client_secret = reg_result.get("client_secret", "")

    if not client_id:
        client_id = "autogpt-platform"

    # Step 4: Store state token with OAuth metadata for the callback
    scopes = (protected_resource or {}).get("scopes_supported") or metadata.get(
        "scopes_supported", []
    )
    state_token, code_challenge = await creds_manager.store.store_state_token(
        user_id,
        ProviderName.MCP.value,
        scopes,
        state_metadata={
            "authorize_url": authorize_url,
            "token_url": token_url,
            "revoke_url": revoke_url,
            "resource_url": resource_url,
            "server_url": server_url,
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )

    # Step 5: Build and return the login URL
    handler = MCPOAuthHandler(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        authorize_url=authorize_url,
        token_url=token_url,
        resource_url=resource_url,
    )
    login_url = handler.get_login_url(
        scopes, state_token, code_challenge=code_challenge
    )

    return MCPOAuthLoginResponse(login_url=login_url, state_token=state_token)


class MCPOAuthCallbackRequest(BaseModel):
    """Request to exchange an OAuth code for tokens."""

    code: str = Field(description="Authorization code from OAuth callback")
    state_token: str = Field(description="State token for CSRF verification")


class MCPOAuthCallbackResponse(BaseModel):
    """Response after successfully storing OAuth credentials."""

    credential_id: str


@router.post(
    "/oauth/callback",
    summary="Exchange OAuth code for MCP tokens",
)
async def mcp_oauth_callback(
    request: MCPOAuthCallbackRequest,
    user_id: Annotated[str, Security(get_user_id)],
) -> CredentialsMetaResponse:
    """
    Exchange the authorization code for tokens and store the credential.

    The frontend calls this after receiving the OAuth code from the popup.
    On success, subsequent ``/discover-tools`` calls for the same server URL
    will automatically use the stored credential.
    """
    valid_state = await creds_manager.store.verify_state_token(
        user_id, request.state_token, ProviderName.MCP.value
    )
    if not valid_state:
        raise fastapi.HTTPException(
            status_code=400,
            detail="Invalid or expired state token.",
        )

    meta = valid_state.state_metadata
    frontend_base_url = settings.config.frontend_base_url
    if not frontend_base_url:
        raise fastapi.HTTPException(
            status_code=500,
            detail="Frontend base URL is not configured.",
        )
    redirect_uri = f"{frontend_base_url}/auth/integrations/mcp_callback"

    handler = MCPOAuthHandler(
        client_id=meta["client_id"],
        client_secret=meta.get("client_secret", ""),
        redirect_uri=redirect_uri,
        authorize_url=meta["authorize_url"],
        token_url=meta["token_url"],
        revoke_url=meta.get("revoke_url"),
        resource_url=meta.get("resource_url"),
    )

    try:
        credentials = await handler.exchange_code_for_tokens(
            request.code, valid_state.scopes, valid_state.code_verifier
        )
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"OAuth token exchange failed: {e}",
        )

    # Enrich credential metadata for future lookup and token refresh
    if credentials.metadata is None:
        credentials.metadata = {}
    credentials.metadata["mcp_server_url"] = meta["server_url"]
    credentials.metadata["mcp_client_id"] = meta["client_id"]
    credentials.metadata["mcp_client_secret"] = meta.get("client_secret", "")
    credentials.metadata["mcp_token_url"] = meta["token_url"]
    credentials.metadata["mcp_resource_url"] = meta.get("resource_url", "")

    hostname = server_host(meta["server_url"])
    credentials.title = f"MCP: {hostname}"

    # Remove old MCP credentials for the same server to prevent stale token buildup.
    try:
        old_creds = await creds_manager.store.get_creds_by_provider(
            user_id, ProviderName.MCP.value
        )
        for old in old_creds:
            if (
                isinstance(old, OAuth2Credentials)
                and (old.metadata or {}).get("mcp_server_url") == meta["server_url"]
            ):
                await creds_manager.store.delete_creds_by_id(user_id, old.id)
                logger.info(
                    "Removed old MCP credential %s for %s",
                    old.id,
                    server_host(meta["server_url"]),
                )
    except Exception:
        logger.debug("Could not clean up old MCP credentials", exc_info=True)

    await creds_manager.create(user_id, credentials)

    return to_meta_response(credentials)


# ======================== Manual Authentication ======================== #


class MCPStoreTokenRequest(BaseModel):
    """Request to store a manual Basic/Bearer credential or Authorization header."""

    server_url: str = Field(
        description="MCP server URL the credential authenticates against"
    )
    token: SecretStr = Field(
        min_length=1,
        max_length=_MAX_CREDENTIAL_LENGTH,
        description=(
            "Bare Bearer token, Basic/Bearer value, or complete Authorization header"
        ),
    )


@router.post(
    "/token",
    # The summary is pinned so the generated client keeps its current method
    # name; the description is where Basic support is documented.
    summary="Store a bearer token for an MCP server",
    description=(
        "Store a manually entered MCP credential. Accepts a bare token "
        "(sent as Bearer, unchanged from before), an explicit `Basic <value>` "
        "or `Bearer <value>`, or a complete `Authorization:` header."
    ),
)
async def mcp_store_token(
    request: MCPStoreTokenRequest,
    user_id: Annotated[str, Security(get_user_id)],
) -> CredentialsMetaResponse:
    """
    Store a Basic/Bearer credential or complete Authorization header for an MCP server.

    Used by the Copilot MCPSetupCard when the server doesn't support the MCP
    OAuth discovery flow (returns 400 from /oauth/login).  Subsequent
    ``run_mcp_tool`` calls will automatically pick up the credential via
    ``_auto_lookup_credential``.
    """
    try:
        authorization = normalize_mcp_authorization(request.token.get_secret_value())
    except ValueError as e:
        raise fastapi.HTTPException(status_code=422, detail=str(e)) from e

    # Validate URL to prevent SSRF — blocks loopback and private IP ranges.
    try:
        await validate_url_host(request.server_url)
    except ValueError as e:
        raise fastapi.HTTPException(status_code=400, detail=f"Invalid server URL: {e}")

    # Normalize URL so trailing-slash variants match existing credentials.
    server_url = normalize_mcp_url(request.server_url)
    hostname = server_host(server_url)

    # Reuse an existing *manual* credential ID so saved graphs keep resolving
    # their credential references after a token rotation.  An OAuth row for the
    # same server is never rewritten in place: that would keep its ID and
    # ``type="oauth2"`` while destroying the refresh token and the
    # ``mcp_client_id``/``mcp_token_url`` metadata behind it, so a saved graph
    # would silently start running on a pasted static secret.
    manual_credentials: list[OAuth2Credentials] = []
    try:
        old_creds = await creds_manager.store.get_creds_by_provider(
            user_id, ProviderName.MCP.value
        )
        for old in old_creds:
            if (
                not isinstance(old, OAuth2Credentials)
                or old.is_managed
                or normalize_mcp_url((old.metadata or {}).get("mcp_server_url", ""))
                != server_url
            ):
                continue
            if is_manual_mcp_credential(old):
                manual_credentials.append(old)
    except Exception as e:
        logger.exception("Could not query existing MCP credentials")
        raise fastapi.HTTPException(
            status_code=503,
            detail="Could not safely update the MCP credential. Please try again.",
        ) from e

    auth_scheme = authorization.split(" ", 1)[0].lower()
    metadata = {"mcp_server_url": server_url, "mcp_auth_scheme": auth_scheme}
    # Every write below is a read-modify-write of the user's whole credential
    # set, so touch exactly one row: rotate one manual credential and drop the
    # redundant duplicates, rather than updating each in turn.
    #
    # `get_creds_by_provider` promises no ordering, so pick the survivor by an
    # explicit key instead of by position.
    manual_credentials.sort(key=lambda cred: cred.id)
    survivor = manual_credentials[-1] if manual_credentials else None
    superseded_ids = [old.id for old in manual_credentials[:-1]]
    if survivor is not None:
        credentials = survivor.model_copy(
            update={
                "title": f"MCP: {hostname}",
                "username": None,
                "access_token": SecretStr(authorization),
                "access_token_expires_at": None,
                "scopes": [],
                "metadata": metadata,
            }
        )
        await creds_manager.update(user_id, credentials)
    else:
        credentials = OAuth2Credentials(
            provider=ProviderName.MCP.value,
            title=f"MCP: {hostname}",
            access_token=SecretStr(authorization),
            scopes=[],
            metadata=metadata,
        )
        await creds_manager.create(user_id, credentials)

    # Redundant *manual* duplicates only.  An OAuth row for the same server is
    # deliberately left alone: neither `creds_manager.delete` nor
    # `delete_acquired` calls `handler.revoke_tokens` — that lives in
    # `DELETE /credentials` in the integrations router — so deleting one here
    # would drop the row while leaving a live refresh token at the provider
    # with nothing left to revoke it with, and would break every saved graph
    # node bound to its ID.  `auto_lookup_mcp_credential` prefers the manual
    # credential over a surviving OAuth row, so the pasted one is the one that
    # gets sent; disconnecting the OAuth grant stays an explicit user action
    # through the route that actually revokes it.
    #
    # Deleted only after the new credential is safely stored, so a failed write
    # leaves the user with their previous credential rather than none at all,
    # and through `creds_manager.delete` rather than `store.delete_creds_by_id`
    # because the former takes the per-credential lock and fires the
    # credentials-changed hook that evicts cached provider tokens.
    for old_id in superseded_ids:
        try:
            await creds_manager.delete(user_id, old_id)
        except Exception:
            logger.debug("Could not clean up superseded MCP credential", exc_info=True)

    return to_meta_response(credentials)


# ======================== Helpers ======================== #


async def _register_mcp_client(
    registration_endpoint: str,
    redirect_uri: str,
    server_url: str,
) -> dict[str, Any] | None:
    """Attempt Dynamic Client Registration (RFC 7591) with an MCP auth server."""
    try:
        response = await Requests(raise_for_status=True).post(
            registration_endpoint,
            json={
                "client_name": "AutoGPT Platform",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "client_secret_post",
            },
        )
        data = response.json()
        if isinstance(data, dict) and "client_id" in data:
            return data
        return None
    except Exception as e:
        logger.warning(
            "Dynamic client registration failed for %s: %s", server_host(server_url), e
        )
        return None
