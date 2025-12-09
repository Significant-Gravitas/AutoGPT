"""
Integration Connect popup endpoints.

Implements the popup flow for external applications to connect integrations
on behalf of users through AutoGPT's Credential Broker.

Flow:
1. External app opens popup to /connect/{provider}
2. User sees consent page with existing credentials or option to connect new
3. User approves, grant is created
4. Popup sends postMessage with grant_id back to opener
"""

import html
import logging
from typing import Annotated, Optional

from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, Form, Query, Request, Security
from fastapi.responses import HTMLResponse, RedirectResponse
from prisma.enums import CredentialGrantPermission

from backend.data.credential_grants import (
    create_credential_grant,
    get_grant_by_credential_and_client,
    update_grant_scopes,
)
from backend.data.db import prisma
from backend.data.integration_scopes import (
    INTEGRATION_SCOPE_DESCRIPTIONS,
    get_provider_for_scope,
    get_provider_scopes,
    validate_integration_scopes,
)
from backend.data.model import OAuth2Credentials
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.oauth import HANDLERS_BY_NAME
from backend.integrations.providers import ProviderName
from backend.server.integrations.connect_security import (
    consume_connect_continuation,
    consume_connect_state,
    create_post_message_data,
    store_connect_continuation,
    store_connect_state,
    validate_nonce,
    validate_redirect_origin,
)
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

connect_router = APIRouter(prefix="/connect", tags=["integration-connect"])

creds_manager = IntegrationCredentialsManager()
settings = Settings()


async def _create_or_update_grant(
    user_id: str,
    credential_id: str,
    client_db_id: str,
    provider: str,
    requested_scopes: list[str],
) -> str:
    """
    Create a new credential grant or update existing one with merged scopes.

    Args:
        user_id: User who owns the credential
        credential_id: ID of the credential to grant access to
        client_db_id: Database UUID of the OAuth client
        provider: Integration provider name
        requested_scopes: Scopes being requested

    Returns:
        The grant ID (either existing or newly created)
    """
    existing_grant = await get_grant_by_credential_and_client(
        user_id=user_id,
        credential_id=credential_id,
        client_id=client_db_id,
    )

    if existing_grant:
        # Update scopes if needed (merge with existing)
        merged_scopes = list(set(existing_grant.grantedScopes) | set(requested_scopes))
        if set(merged_scopes) != set(existing_grant.grantedScopes):
            await update_grant_scopes(existing_grant.id, merged_scopes)
        return existing_grant.id

    # Create new grant
    grant = await create_credential_grant(
        user_id=user_id,
        client_id=client_db_id,
        credential_id=credential_id,
        provider=provider,
        granted_scopes=requested_scopes,
        permissions=[
            CredentialGrantPermission.USE,
            CredentialGrantPermission.DELETE,
        ],
    )
    return grant.id


def _base_styles() -> str:
    """Common CSS styles for connect pages."""
    return """
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            color: #e4e4e7;
        }
        .container {
            background: #27272a;
            border-radius: 16px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            max-width: 450px;
            width: 100%;
            padding: 32px;
        }
        h1 {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 8px;
            text-align: center;
        }
        .subtitle {
            color: #a1a1aa;
            font-size: 14px;
            text-align: center;
            margin-bottom: 24px;
        }
        .divider {
            height: 1px;
            background: #3f3f46;
            margin: 20px 0;
        }
        .section-title {
            font-size: 14px;
            font-weight: 500;
            color: #a1a1aa;
            margin-bottom: 12px;
        }
        .credential-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            border: 1px solid #3f3f46;
            border-radius: 8px;
            cursor: pointer;
            margin-bottom: 8px;
            transition: all 0.2s;
        }
        .credential-item:hover {
            border-color: #22d3ee;
            background: rgba(34, 211, 238, 0.1);
        }
        .credential-item.selected {
            border-color: #22d3ee;
            background: rgba(34, 211, 238, 0.15);
        }
        .credential-item input[type="radio"] {
            display: none;
        }
        .credential-info {
            flex: 1;
        }
        .credential-title {
            font-size: 14px;
            font-weight: 500;
        }
        .credential-meta {
            font-size: 12px;
            color: #71717a;
        }
        .scope-item {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            padding: 8px 0;
            font-size: 14px;
        }
        .scope-icon {
            color: #22d3ee;
            flex-shrink: 0;
        }
        .buttons {
            display: flex;
            gap: 12px;
            margin-top: 24px;
        }
        .btn {
            flex: 1;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border: none;
            transition: all 0.2s;
        }
        .btn-cancel {
            background: #3f3f46;
            color: #e4e4e7;
        }
        .btn-cancel:hover {
            background: #52525b;
        }
        .btn-primary {
            background: #22d3ee;
            color: #0f172a;
        }
        .btn-primary:hover {
            background: #06b6d4;
        }
        .btn-connect {
            background: #3b82f6;
            color: white;
        }
        .btn-connect:hover {
            background: #2563eb;
        }
        .error-message {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 8px;
            padding: 12px;
            color: #ef4444;
            font-size: 14px;
            text-align: center;
        }
        .app-name {
            color: #22d3ee;
            font-weight: 600;
        }
        .provider-badge {
            display: inline-block;
            background: #3f3f46;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            text-transform: capitalize;
        }
    """


def _render_connect_page(
    client_name: str,
    provider: str,
    scopes: list[str],
    credentials: list[OAuth2Credentials],
    connect_token: str,
    action_url: str,
) -> str:
    """Render the connect consent page."""
    # Build scopes HTML
    scopes_html = ""
    for scope in scopes:
        description = INTEGRATION_SCOPE_DESCRIPTIONS.get(scope, scope)
        scopes_html += f"""
            <div class="scope-item">
                <span class="scope-icon">&#10003;</span>
                <span>{description}</span>
            </div>
        """

    # Build credentials selection HTML
    creds_html = ""
    if credentials:
        creds_html = '<div class="section-title">Select an existing credential:</div>'
        for i, cred in enumerate(credentials):
            checked = "checked" if i == 0 else ""
            selected = "selected" if i == 0 else ""
            creds_html += f"""
                <label class="credential-item {selected}" onclick="selectCredential(this)">
                    <input type="radio" name="credential_id" value="{cred.id}" {checked}>
                    <div class="credential-info">
                        <div class="credential-title">{cred.title or cred.username or 'Credential'}</div>
                        <div class="credential-meta">{cred.username or ''}</div>
                    </div>
                </label>
            """
        creds_html += '<div class="divider"></div>'

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Connect {provider.title()} - AutoGPT</title>
        <style>{_base_styles()}</style>
        <script>
            function selectCredential(element) {{
                document.querySelectorAll('.credential-item').forEach(el => el.classList.remove('selected'));
                element.classList.add('selected');
                element.querySelector('input[type="radio"]').checked = true;
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Connect <span class="provider-badge">{provider}</span></h1>
            <p class="subtitle">
                <span class="app-name">{client_name}</span> wants to use your {provider.title()} integration
            </p>

            <div class="divider"></div>

            <div class="section-title">This will allow {client_name} to:</div>
            {scopes_html}

            <div class="divider"></div>

            <form method="POST" action="{action_url}">
                <input type="hidden" name="connect_token" value="{connect_token}">

                {creds_html}

                {'''<div class="section-title">Or connect a new account:</div>
                <button type="submit" name="action" value="connect_new" class="btn btn-connect" style="width: 100%; margin-bottom: 16px;">
                    Connect New {0} Account
                </button>'''.format(provider.title()) if credentials else '''
                <p class="subtitle" style="margin-bottom: 16px;">
                    You don't have any {0} credentials yet.
                </p>
                <button type="submit" name="action" value="connect_new" class="btn btn-connect" style="width: 100%; margin-bottom: 16px;">
                    Connect {0} Account
                </button>
                '''.format(provider.title())}

                <div class="buttons">
                    <button type="submit" name="action" value="deny" class="btn btn-cancel">
                        Cancel
                    </button>
                    {'''<button type="submit" name="action" value="approve" class="btn btn-primary">
                        Approve
                    </button>''' if credentials else ''}
                </div>
            </form>
        </div>
    </body>
    </html>
    """


def _render_error_page(error: str, error_description: str) -> str:
    """Render an error page."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Connection Error - AutoGPT</title>
        <style>{_base_styles()}</style>
    </head>
    <body>
        <div class="container">
            <h1 style="color: #ef4444;">Connection Failed</h1>
            <div class="error-message" style="margin-top: 24px;">
                {error_description}
            </div>
            <p class="subtitle" style="margin-top: 16px;">
                Error code: {error}
            </p>
        </div>
    </body>
    </html>
    """


def _render_result_page(
    success: bool,
    redirect_origin: str,
    post_message_data: dict,
) -> str:
    """Render a result page that sends postMessage to opener."""
    import json

    status_class = "color: #22c55e;" if success else "color: #ef4444;"
    status_text = "Connected Successfully!" if success else "Connection Failed"
    message = (
        "You can close this window."
        if success
        else post_message_data.get("error_description", "An error occurred")
    )

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{'Connected' if success else 'Error'} - AutoGPT</title>
        <style>{_base_styles()}</style>
    </head>
    <body>
        <div class="container">
            <h1 style="{status_class}">{status_text}</h1>
            <p class="subtitle" style="margin-top: 16px;">
                {message}
            </p>
            <p class="subtitle" style="margin-top: 8px; font-size: 12px;">
                This window will close automatically...
            </p>
        </div>
        <script>
            (function() {{
                var targetOrigin = {json.dumps(redirect_origin)};
                var message = {json.dumps(post_message_data)};
                if (window.opener) {{
                    window.opener.postMessage(message, targetOrigin);
                    setTimeout(function() {{ window.close(); }}, 1500);
                }}
            }})();
        </script>
    </body>
    </html>
    """


@connect_router.get("/{provider}", response_model=None)
async def connect_page(
    provider: ProviderName,
    client_id: Annotated[str, Query(description="OAuth client ID")],
    scopes: Annotated[str, Query(description="Comma-separated integration scopes")],
    nonce: Annotated[str, Query(description="Nonce for replay protection")],
    redirect_origin: Annotated[str, Query(description="Origin for postMessage")],
    user_id: Annotated[str, Security(get_user_id)],
) -> HTMLResponse:
    """
    Render the connect consent page.

    This page allows users to select an existing credential or connect a new one
    for use by an external application.
    """
    # Validate client
    client = await prisma.oauthclient.find_unique(where={"clientId": client_id})
    if not client:
        return HTMLResponse(
            _render_error_page("invalid_client", "Unknown application"),
            status_code=400,
        )

    if client.status.value != "ACTIVE":
        return HTMLResponse(
            _render_error_page("invalid_client", "Application is not active"),
            status_code=400,
        )

    # Validate redirect origin
    if not validate_redirect_origin(redirect_origin, client):
        return HTMLResponse(
            _render_error_page(
                "invalid_request", "Invalid redirect origin for this application"
            ),
            status_code=400,
        )

    # Validate nonce
    if not await validate_nonce(client_id, nonce):
        return HTMLResponse(
            _render_error_page("invalid_request", "Nonce has already been used"),
            status_code=400,
        )

    # Parse and validate scopes
    requested_scopes = [s.strip() for s in scopes.split(",") if s.strip()]
    valid, invalid = validate_integration_scopes(requested_scopes)

    if not valid:
        # HTML escape user input to prevent XSS
        escaped_invalid = html.escape(", ".join(invalid))
        return HTMLResponse(
            _render_error_page(
                "invalid_scope", f"Invalid scopes requested: {escaped_invalid}"
            ),
            status_code=400,
        )

    # Verify all scopes are for the requested provider
    for scope in requested_scopes:
        scope_provider = get_provider_for_scope(scope)
        if scope_provider != provider:
            # HTML escape user input to prevent XSS
            escaped_scope = html.escape(scope)
            return HTMLResponse(
                _render_error_page(
                    "invalid_scope",
                    f"Scope '{escaped_scope}' is not for provider '{provider.value}'",
                ),
                status_code=400,
            )

    # Get user's existing credentials for this provider
    user_credentials = await creds_manager.store.get_creds_by_provider(
        user_id, provider
    )
    oauth_credentials = [
        c for c in user_credentials if isinstance(c, OAuth2Credentials)
    ]

    # Store connect state
    connect_token = await store_connect_state(
        user_id=user_id,
        client_id=client_id,
        provider=provider.value,
        requested_scopes=requested_scopes,
        redirect_origin=redirect_origin,
        nonce=nonce,
    )

    return HTMLResponse(
        _render_connect_page(
            client_name=client.name,
            provider=provider.value,
            scopes=requested_scopes,
            credentials=oauth_credentials,
            connect_token=connect_token,
            action_url=f"/connect/{provider.value}/approve",
        )
    )


@connect_router.post("/{provider}/approve", response_model=None)
async def approve_connect(
    provider: ProviderName,
    request: Request,
    connect_token: Annotated[str, Form()],
    action: Annotated[str, Form()],
    credential_id: Annotated[Optional[str], Form()] = None,
    user_id: Annotated[str, Security(get_user_id)] = "",
) -> HTMLResponse | RedirectResponse:
    """
    Process the connect form submission.

    Creates a credential grant and returns a page that sends postMessage to opener.
    """
    # Consume state (one-time use)
    state = await consume_connect_state(connect_token)
    if not state:
        return HTMLResponse(
            _render_error_page("invalid_request", "Invalid or expired connect session"),
            status_code=400,
        )

    # Verify user
    if state.user_id != user_id:
        return HTMLResponse(
            _render_error_page("access_denied", "User mismatch"),
            status_code=403,
        )

    redirect_origin = state.redirect_origin
    nonce = state.nonce
    requested_scopes = state.requested_scopes
    client_id = state.client_id

    # Handle denial
    if action == "deny":
        post_data = create_post_message_data(
            success=False,
            error="access_denied",
            error_description="User denied the connection request",
            nonce=nonce,
        )
        return HTMLResponse(_render_result_page(False, redirect_origin, post_data))

    # Handle connect new - redirect to OAuth login
    if action == "connect_new":
        # Get client database ID for continuation state
        client = await prisma.oauthclient.find_unique(where={"clientId": client_id})
        if not client:
            post_data = create_post_message_data(
                success=False,
                error="invalid_client",
                error_description="Client not found",
                nonce=nonce,
            )
            return HTMLResponse(_render_result_page(False, redirect_origin, post_data))

        # Get OAuth handler for this provider
        handler = _get_provider_oauth_handler(request, provider)
        if not handler:
            post_data = create_post_message_data(
                success=False,
                error="unsupported_provider",
                error_description=f"Provider '{provider.value}' does not support OAuth",
                nonce=nonce,
            )
            return HTMLResponse(_render_result_page(False, redirect_origin, post_data))

        # Store continuation state for after OAuth completes
        continuation_token = await store_connect_continuation(
            user_id=user_id,
            client_id=client_id,
            client_db_id=client.id,
            provider=provider.value,
            requested_scopes=requested_scopes,
            redirect_origin=redirect_origin,
            nonce=nonce,
        )

        # Convert integration scopes to provider OAuth scopes
        provider_scopes = get_provider_scopes(provider, requested_scopes)

        # Store OAuth state with continuation token in metadata
        state_token, code_challenge = await creds_manager.store.store_state_token(
            user_id=user_id,
            provider=provider.value,
            scopes=provider_scopes,
            use_pkce=True,
            state_metadata={"connect_continuation": continuation_token},
        )

        # Build OAuth URL and redirect
        login_url = handler.get_login_url(
            provider_scopes, state_token, code_challenge=code_challenge
        )

        logger.info(
            f"Redirecting to OAuth for connect_new: provider={provider.value}, "
            f"user={user_id}, client={client_id}"
        )

        return RedirectResponse(url=login_url, status_code=302)

    # Handle approval with existing credential
    if action == "approve":
        if not credential_id:
            post_data = create_post_message_data(
                success=False,
                error="invalid_request",
                error_description="No credential selected",
                nonce=nonce,
            )
            return HTMLResponse(_render_result_page(False, redirect_origin, post_data))

        # Verify credential belongs to user and provider
        credential = await creds_manager.get(user_id, credential_id)
        if not credential:
            post_data = create_post_message_data(
                success=False,
                error="invalid_request",
                error_description="Credential not found",
                nonce=nonce,
            )
            return HTMLResponse(_render_result_page(False, redirect_origin, post_data))

        if credential.provider != provider.value:
            post_data = create_post_message_data(
                success=False,
                error="invalid_request",
                error_description="Credential provider mismatch",
                nonce=nonce,
            )
            return HTMLResponse(_render_result_page(False, redirect_origin, post_data))

        # Get client database ID
        client = await prisma.oauthclient.find_unique(where={"clientId": client_id})
        if not client:
            post_data = create_post_message_data(
                success=False,
                error="invalid_client",
                error_description="Client not found",
                nonce=nonce,
            )
            return HTMLResponse(_render_result_page(False, redirect_origin, post_data))

        # Create or update grant
        grant_id = await _create_or_update_grant(
            user_id=user_id,
            credential_id=credential_id,
            client_db_id=client.id,
            provider=provider.value,
            requested_scopes=requested_scopes,
        )

        post_data = create_post_message_data(
            success=True,
            grant_id=grant_id,
            credential_id=credential_id,
            provider=provider.value,
            nonce=nonce,
        )
        return HTMLResponse(_render_result_page(True, redirect_origin, post_data))

    # Unknown action
    post_data = create_post_message_data(
        success=False,
        error="invalid_request",
        error_description="Unknown action",
        nonce=nonce,
    )
    return HTMLResponse(_render_result_page(False, redirect_origin, post_data))


@connect_router.get("/{provider}/callback", response_model=None)
async def connect_oauth_callback(
    provider: ProviderName,
    request: Request,
    code: str,
    state: str,
    user_id: Annotated[str, Security(get_user_id)],
) -> HTMLResponse:
    """
    Handle OAuth callback after user authorizes a new connection.

    This endpoint is called after the OAuth provider redirects back with an
    authorization code. It exchanges the code for tokens, creates the credential,
    creates the grant, and returns a page that sends postMessage to the opener.
    """
    # Get OAuth handler
    handler = _get_provider_oauth_handler(request, provider)
    if not handler:
        return HTMLResponse(
            _render_error_page(
                "unsupported_provider",
                f"Provider '{provider.value}' does not support OAuth",
            ),
            status_code=400,
        )

    # Verify the state token and get the associated state
    valid_state = await creds_manager.store.verify_state_token(
        user_id, state, provider.value
    )

    if not valid_state:
        return HTMLResponse(
            _render_error_page(
                "invalid_state",
                "Invalid or expired OAuth state. Please try again.",
            ),
            status_code=400,
        )

    # Check for continuation token in state metadata
    continuation_token = valid_state.state_metadata.get("connect_continuation")
    if not continuation_token:
        return HTMLResponse(
            _render_error_page(
                "invalid_request",
                "Missing continuation token. Please try again.",
            ),
            status_code=400,
        )

    # Get continuation state
    continuation = await consume_connect_continuation(continuation_token)
    if not continuation:
        return HTMLResponse(
            _render_error_page(
                "invalid_request",
                "Invalid or expired continuation state. Please try again.",
            ),
            status_code=400,
        )

    # Verify user matches
    if continuation.user_id != user_id:
        return HTMLResponse(
            _render_error_page("access_denied", "User mismatch"),
            status_code=403,
        )

    redirect_origin = continuation.redirect_origin
    nonce = continuation.nonce
    requested_scopes = continuation.requested_scopes
    client_db_id = continuation.client_db_id

    try:
        # Handle default scopes
        scopes = handler.handle_default_scopes(valid_state.scopes)

        # Exchange code for tokens
        credentials = await handler.exchange_code_for_tokens(
            code, scopes, valid_state.code_verifier
        )

        # Linear returns scopes as a single string with spaces
        if len(credentials.scopes) == 1 and " " in credentials.scopes[0]:
            credentials.scopes = credentials.scopes[0].split(" ")

        # Store the new credentials
        await creds_manager.create(user_id, credentials)

        logger.info(
            f"Created new credential via connect flow: provider={provider.value}, "
            f"user={user_id}, credential={credentials.id}"
        )

    except Exception as e:
        logger.error(f"OAuth token exchange failed: {e}")
        post_data = create_post_message_data(
            success=False,
            error="oauth_error",
            error_description=f"Failed to complete OAuth: {str(e)}",
            nonce=nonce,
        )
        return HTMLResponse(_render_result_page(False, redirect_origin, post_data))

    # Create the credential grant
    try:
        grant_id = await _create_or_update_grant(
            user_id=user_id,
            credential_id=credentials.id,
            client_db_id=client_db_id,
            provider=provider.value,
            requested_scopes=requested_scopes,
        )

        logger.info(
            f"Created grant via connect flow: grant={grant_id}, "
            f"credential={credentials.id}, client={client_db_id}"
        )

        post_data = create_post_message_data(
            success=True,
            grant_id=grant_id,
            credential_id=credentials.id,
            provider=provider.value,
            nonce=nonce,
        )
        return HTMLResponse(_render_result_page(True, redirect_origin, post_data))

    except Exception as e:
        logger.error(f"Failed to create grant: {e}")
        post_data = create_post_message_data(
            success=False,
            error="grant_error",
            error_description=f"Failed to create grant: {str(e)}",
            nonce=nonce,
        )
        return HTMLResponse(_render_result_page(False, redirect_origin, post_data))


def _get_provider_oauth_handler(request: Request, provider_name: ProviderName):
    """Get the OAuth handler for a provider."""
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        pass

    from backend.integrations.oauth import CREDENTIALS_BY_PROVIDER

    # Ensure blocks are loaded so SDK providers are available
    try:
        from backend.blocks import load_all_blocks

        load_all_blocks()  # This is cached, so it only runs once
    except Exception as e:
        logger.warning(f"Failed to load blocks: {e}")

    # Convert provider_name to string for lookup
    provider_key = (
        provider_name.value if hasattr(provider_name, "value") else str(provider_name)
    )

    if provider_key not in HANDLERS_BY_NAME:
        return None

    # Check if this provider has custom OAuth credentials
    oauth_credentials = CREDENTIALS_BY_PROVIDER.get(provider_key)

    if oauth_credentials and not oauth_credentials.use_secrets:
        # SDK provider with custom env vars
        import os

        client_id = (
            os.getenv(oauth_credentials.client_id_env_var)
            if oauth_credentials.client_id_env_var
            else None
        )
        client_secret = (
            os.getenv(oauth_credentials.client_secret_env_var)
            if oauth_credentials.client_secret_env_var
            else None
        )
    else:
        # Original provider using settings.secrets
        client_id = getattr(settings.secrets, f"{provider_name.value}_client_id", None)
        client_secret = getattr(
            settings.secrets, f"{provider_name.value}_client_secret", None
        )

    if not (client_id and client_secret):
        logger.warning(
            f"OAuth integration not configured for provider {provider_name.value}"
        )
        return None

    handler_class = HANDLERS_BY_NAME[provider_key]
    frontend_base_url = settings.config.frontend_base_url

    if not frontend_base_url:
        logger.error("Frontend base URL is not configured")
        return None

    return handler_class(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=f"{frontend_base_url}/auth/integrations/oauth_callback",
    )
