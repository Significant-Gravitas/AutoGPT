"""
MCP (Model Context Protocol) API routes.

Provides endpoints for MCP tool discovery and OAuth authentication so the
frontend can list available tools on an MCP server before placing a block.
"""

import asyncio
import logging
from typing import Annotated, Any
from urllib.parse import urlparse

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
from backend.util.request import (
    AUTH_STATUS_CODES,
    CREDENTIAL_REJECTED_STATUS_CODES,
    HTTPClientError,
    HTTPServerError,
    Requests,
    validate_url_host,
)
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()
router = fastapi.APIRouter(tags=["mcp"])
creds_manager = IntegrationCredentialsManager()

# Verifying a token is best-effort; it must never outlast a user's patience.
_PROBE_TIMEOUT_SECONDS = 10
_PROBE_CLOSE_TIMEOUT_SECONDS = 5


# ====================== Tool Discovery ====================== #


# Bounds the per-character scan in `normalize_mcp_authorization`.
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
        if e.status_code in AUTH_STATUS_CODES:
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
        # Release any legacy session; a no-op on stateless servers.
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
    # The issuer the metadata document must declare, decided by which
    # well-known URL answered; its ``issuer`` is only trusted when it matches
    # (RFC 8414 §3.3).
    expected_issuer = server_url

    if protected_resource and protected_resource.get("authorization_servers"):
        auth_server_url = protected_resource["authorization_servers"][0]
        resource_url = _trusted_resource(protected_resource.get("resource"), server_url)

        # Validate the auth server URL from metadata to prevent SSRF.
        try:
            await validate_url_host(auth_server_url)
        except ValueError as e:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"Invalid authorization server URL in metadata: {e}",
            )

        # Step 2a: Discover auth-server metadata (RFC 8414)
        discovered = await client.discover_auth_server_metadata(auth_server_url)
        if discovered:
            metadata, expected_issuer = discovered
    else:
        # Fallback: Some MCP servers (e.g. Linear) are their own auth server
        # and serve OAuth metadata directly without protected-resource metadata.
        # Don't assume a resource_url — omitting it lets the auth server choose
        # the correct audience for the token (RFC 8707 resource is optional).
        resource_url = None
        discovered = await client.discover_auth_server_metadata(server_url)
        if discovered:
            metadata, expected_issuer = discovered

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
    # RFC 8414 issuer identifier: validated against the ``iss``
    # authorization-response parameter (RFC 9207) on callback, and recorded
    # on the credential so it stays bound to the authorization server that
    # issued it.  Servers that advertise ``iss`` support must send it.
    issuer = _validated_issuer(metadata, expected_issuer)
    iss_required = bool(issuer) and (
        metadata.get("authorization_response_iss_parameter_supported") is True
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
            "issuer": issuer,
            "iss_required": iss_required,
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
    iss: str | None = Field(
        default=None,
        description="Issuer identifier from the authorization response (RFC 9207). "
        "Must match the authorization server discovered at login; required when "
        "that server advertises `authorization_response_iss_parameter_supported`.",
    )


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
    expected_issuer = meta.get("issuer") or ""
    # RFC 9207 / MCP 2026-07-28: ``iss`` must match the issuer discovered at
    # login, otherwise the code may come from a mix-up attack.  A server that
    # advertises ``iss`` support must send it; older servers may omit it.
    if request.iss is None:
        if meta.get("iss_required"):
            raise fastapi.HTTPException(
                status_code=400,
                detail="Authorization response is missing the issuer identifier "
                "the authorization server advertised it would send.",
            )
    elif expected_issuer and request.iss != expected_issuer:
        raise fastapi.HTTPException(
            status_code=400,
            detail="Authorization response issuer does not match the "
            "authorization server this login was started with.",
        )

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
    credentials.metadata["mcp_issuer"] = expected_issuer

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
    # The summary names the generated client method, so it stays as is.
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

    # Normalize URL so trailing-slash and scheme-less variants match existing
    # credentials — and so the value stored below is the one every lookup path
    # re-derives from the same user input.
    server_url = normalize_mcp_url(request.server_url)

    hostname = server_host(server_url)

    # ``validate_url_host`` permits http:// for MCP servers generally, but a
    # credential travels in a header and must not go out in cleartext.
    if not server_url.lower().startswith("https://"):
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"{hostname} must be reached over https:// — a credential "
            "cannot be sent over an unencrypted connection.",
        )

    # A 2xx from this endpoint is what turns the setup card's pill green, so
    # it has to mean "this credential authenticates against the server" rather
    # than "a row was written". One ``initialize`` round-trip is the cheapest
    # proof. Only an unambiguous rejection blocks the save: any other outcome
    # says nothing about the credential, and refusing to store would strand
    # the user.
    #
    # Redirects are not followed: a cross-host hop either carries the
    # credential somewhere the user never named, or drops it and earns a 401
    # we would wrongly report as "you mistyped this".
    probe_client = MCPClient(
        server_url, authorization=authorization, follow_redirects=False
    )
    try:
        # MCPClient sets no timeout and no retry ceiling of its own.
        await asyncio.wait_for(
            probe_client.initialize(), timeout=_PROBE_TIMEOUT_SECONDS
        )
    except (HTTPClientError, HTTPServerError) as e:
        if e.status_code in CREDENTIAL_REJECTED_STATUS_CODES:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"{hostname} rejected this credential. "
                "Please check that you copied it correctly and try again.",
            )
        logger.info(
            "Could not verify MCP credential against %s (HTTP %s) — storing anyway",
            hostname,
            e.status_code,
        )
    except Exception as e:
        # No ``exc_info``: the error text embeds a server-controlled body.
        logger.info(
            "Could not verify MCP credential against %s (%s) — storing anyway",
            hostname,
            type(e).__name__,
        )
    finally:
        # Without the DELETE the probe leaks a session row server-side.
        try:
            await asyncio.wait_for(
                probe_client.close(), timeout=_PROBE_CLOSE_TIMEOUT_SECONDS
            )
        except Exception:
            logger.debug("MCP probe close failed for %s", hostname)

    # Rotate the existing manual credential in place so saved graphs keep their
    # credential ID.  OAuth rows are left alone: rewriting one would drop its
    # refresh token, and deleting one here would not revoke it.
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

    # Deleted only after the new credential is stored, and through
    # `creds_manager.delete` so the lock and credentials-changed hook run.
    for old_id in superseded_ids:
        try:
            await creds_manager.delete(user_id, old_id)
        except Exception:
            logger.debug("Could not clean up superseded MCP credential", exc_info=True)

    return to_meta_response(credentials)


# ======================== Helpers ======================== #


_DEFAULT_PORTS = {"http": "80", "https": "443"}


def _origin(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = parsed.netloc.lower()
    # An explicit default port names the same origin as an absent one, so
    # ``https://host:443`` must not read as a different server to ``https://host``.
    default_port = _DEFAULT_PORTS.get(scheme)
    if default_port and host.endswith(f":{default_port}"):
        host = host[: -len(default_port) - 1]
    return scheme, host


def _trusted_resource(resource: Any, server_url: str) -> str:
    """The protected-resource ``resource`` identifier, if it names *server_url*.

    RFC 9728 §3.3 requires it to be the URL the metadata was fetched for.  A
    server naming another origin would have us request a token minted for a
    different API and then send it to itself (a mix-up); in that case the
    server URL is used as the resource indicator instead.
    """
    if not isinstance(resource, str) or not resource:
        return server_url
    if _origin(resource) != _origin(server_url):
        logger.warning(
            "Ignoring resource %r from %s: it names another origin",
            resource,
            server_host(server_url),
        )
        return server_url
    return resource


def _canonical_issuer(url: str) -> str:
    """*url* with the scheme and host lowercased and a trailing slash dropped.

    Scheme and host are case-insensitive (RFC 3986 §3.1, §3.2.2); the path is
    not, so it is compared verbatim.
    """
    scheme, host = _origin(url)
    return f"{scheme}://{host}{urlparse(url).path.rstrip('/')}"


def _validated_issuer(metadata: dict[str, Any], expected_issuer: str) -> str:
    """The metadata's ``issuer``, rejecting the document if it names another.

    RFC 8414 §3.3 requires the issuer in the metadata document to be the URL
    the document was fetched for.  A document declaring someone else's issuer
    is rejected outright rather than having just its issuer dropped: leaving
    the issuer empty would set ``iss_required`` to ``False`` and turn the
    callback's mismatch check into a no-op, so a hostile authorization server
    could disable RFC 9207 mix-up protection by claiming, say,
    ``https://accounts.google.com``.  We either trust this document or we do
    not; trusting its endpoints while discarding its issuer is the worst of
    both.

    A document that declares no issuer at all is a different case: it claims
    nothing, so there is nothing to bind and nothing to disbelieve.  Mix-up
    protection is simply unavailable, which is how servers predating RFC 9207
    behave.
    """
    issuer = metadata.get("issuer")
    if not isinstance(issuer, str) or not issuer:
        return ""
    if _canonical_issuer(issuer) != _canonical_issuer(expected_issuer):
        logger.warning(
            "Rejecting metadata from %s: it declares issuer %r, which names "
            "another server",
            server_host(expected_issuer),
            issuer,
        )
        raise fastapi.HTTPException(
            status_code=400,
            detail="This MCP server's authorization server metadata declares "
            "an issuer that does not match where the metadata was published. "
            "Sign-in was stopped because the server's identity cannot be "
            "verified.",
        )
    return issuer


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
                # Required by MCP 2026-07-28 so OIDC-backed authorization
                # servers apply web-app redirect URI rules.
                "application_type": "web",
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
