"""
MCP (Model Context Protocol) API routes.

Provides endpoints for MCP tool discovery and OAuth authentication so the
frontend can list available tools on an MCP server before placing a block.
"""

import logging
from typing import Annotated, Any
from urllib.parse import urlparse

import fastapi
from autogpt_libs.auth import get_user_id
from fastapi import Security
from pydantic import BaseModel, Field

from backend.api.features.integrations.router import CredentialsMetaResponse
from backend.blocks.mcp.client import MCPClient, MCPClientError
from backend.blocks.mcp.oauth import MCPOAuthHandler
from backend.data.model import OAuth2Credentials
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.providers import ProviderName
from backend.util.request import HTTPClientError, Requests
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()
router = fastapi.APIRouter(tags=["mcp"])
creds_manager = IntegrationCredentialsManager()


# ====================== Tool Discovery ====================== #


class DiscoverToolsRequest(BaseModel):
    """Request to discover tools on an MCP server."""

    server_url: str = Field(description="URL of the MCP server")
    auth_token: str | None = Field(
        default=None,
        description="Optional Bearer token for authenticated MCP servers",
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
    used automatically — no need to pass an explicit auth token.
    """
    auth_token = request.auth_token

    # Auto-use stored MCP credential when no explicit token is provided.
    if not auth_token:
        mcp_creds = await creds_manager.store.get_creds_by_provider(
            user_id, ProviderName.MCP.value
        )
        # Find the freshest credential for this server URL
        best_cred: OAuth2Credentials | None = None
        for cred in mcp_creds:
            if (
                isinstance(cred, OAuth2Credentials)
                and (cred.metadata or {}).get("mcp_server_url") == request.server_url
            ):
                if best_cred is None or (
                    (cred.access_token_expires_at or 0)
                    > (best_cred.access_token_expires_at or 0)
                ):
                    best_cred = cred
        if best_cred:
            # Refresh the token if expired before using it
            best_cred = await creds_manager.refresh_if_needed(user_id, best_cred)
            logger.info(
                f"Using MCP credential {best_cred.id} for {request.server_url}, "
                f"expires_at={best_cred.access_token_expires_at}"
            )
            auth_token = best_cred.access_token.get_secret_value()

    client = MCPClient(request.server_url, auth_token=auth_token)

    try:
        init_result = await client.initialize()
        tools = await client.list_tools()
    except HTTPClientError as e:
        if e.status_code in (401, 403):
            raise fastapi.HTTPException(
                status_code=401,
                detail="This MCP server requires authentication. "
                "Please provide a valid auth token.",
            )
        raise fastapi.HTTPException(status_code=502, detail=str(e))
    except MCPClientError as e:
        raise fastapi.HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=502,
            detail=f"Failed to connect to MCP server: {e}",
        )

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
            or urlparse(request.server_url).hostname
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
    client = MCPClient(request.server_url)

    # Step 1: Discover protected-resource metadata (RFC 9728)
    protected_resource = await client.discover_auth()

    metadata: dict[str, Any] | None = None

    if protected_resource and protected_resource.get("authorization_servers"):
        auth_server_url = protected_resource["authorization_servers"][0]
        resource_url = protected_resource.get("resource", request.server_url)

        # Step 2a: Discover auth-server metadata (RFC 8414)
        metadata = await client.discover_auth_server_metadata(auth_server_url)
    else:
        # Fallback: Some MCP servers (e.g. Linear) are their own auth server
        # and serve OAuth metadata directly without protected-resource metadata.
        # Don't assume a resource_url — omitting it lets the auth server choose
        # the correct audience for the token (RFC 8707 resource is optional).
        resource_url = None
        metadata = await client.discover_auth_server_metadata(request.server_url)

    if (
        not metadata
        or "authorization_endpoint" not in metadata
        or "token_endpoint" not in metadata
    ):
        raise fastapi.HTTPException(
            status_code=400,
            detail="This MCP server does not advertise OAuth support. "
            "You may need to provide an auth token manually.",
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
        reg_result = await _register_mcp_client(
            registration_endpoint, redirect_uri, request.server_url
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
            "server_url": request.server_url,
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

    hostname = urlparse(meta["server_url"]).hostname or meta["server_url"]
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
                    f"Removed old MCP credential {old.id} for {meta['server_url']}"
                )
    except Exception:
        logger.debug("Could not clean up old MCP credentials", exc_info=True)

    await creds_manager.create(user_id, credentials)

    return CredentialsMetaResponse(
        id=credentials.id,
        provider=credentials.provider,
        type=credentials.type,
        title=credentials.title,
        scopes=credentials.scopes,
        username=credentials.username,
        host=credentials.metadata.get("mcp_server_url"),
    )


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
        logger.warning(f"Dynamic client registration failed for {server_url}: {e}")
        return None
