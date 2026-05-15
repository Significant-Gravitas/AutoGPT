import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Annotated, Any, List, Literal

from autogpt_libs.auth import get_user_id
from fastapi import (
    APIRouter,
    Body,
    HTTPException,
    Path,
    Query,
    Request,
    Security,
    status,
)
from pydantic import BaseModel, Field, model_validator
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_502_BAD_GATEWAY

from backend.api.features.library.db import set_preset_webhook, update_preset
from backend.api.features.library.model import LibraryAgentPreset
from backend.data.graph import NodeModel, get_graph, set_node_webhook
from backend.data.integrations import (
    WebhookEvent,
    WebhookWithRelations,
    get_all_webhooks_by_creds,
    get_webhook,
    publish_webhook_event,
    wait_for_webhook_event,
)
from backend.data.model import (
    APIKeyCredentials,
    Credentials,
    CredentialsType,
    HostScopedCredentials,
    OAuth2Credentials,
    is_sdk_default,
)
from backend.data.onboarding import OnboardingStep, complete_onboarding_step
from backend.executor.utils import add_graph_execution
from backend.integrations.ayrshare import AyrshareClient, SocialPlatform
from backend.integrations.credentials_store import (
    is_system_credential,
    provider_matches,
)
from backend.integrations.creds_manager import (
    IntegrationCredentialsManager,
    create_mcp_oauth_handler,
)
from backend.integrations.managed_credentials import (
    ensure_managed_credential,
    ensure_managed_credentials,
)
from backend.integrations.managed_providers.ayrshare import AyrshareManagedProvider
from backend.integrations.managed_providers.ayrshare import (
    settings_available as ayrshare_settings_available,
)
from backend.integrations.oauth import CREDENTIALS_BY_PROVIDER, HANDLERS_BY_NAME
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks import get_webhook_manager
from backend.util.exceptions import (
    GraphNotInLibraryError,
    MissingConfigError,
    NeedConfirmation,
    NotFoundError,
)
from backend.util.settings import Settings

from .models import (
    ProviderConstants,
    ProviderMetadata,
    ProviderNamesResponse,
    get_all_provider_names,
    get_provider_description,
    get_supported_auth_types,
)

if TYPE_CHECKING:
    from backend.integrations.oauth import BaseOAuthHandler

logger = logging.getLogger(__name__)
settings = Settings()
router = APIRouter()

creds_manager = IntegrationCredentialsManager()


class LoginResponse(BaseModel):
    login_url: str
    state_token: str


@router.get("/{provider}/login", summary="Initiate OAuth flow")
async def login(
    provider: Annotated[
        ProviderName, Path(title="The provider to initiate an OAuth flow for")
    ],
    user_id: Annotated[str, Security(get_user_id)],
    request: Request,
    scopes: Annotated[
        str, Query(title="Comma-separated list of authorization scopes")
    ] = "",
    credential_id: Annotated[
        str | None,
        Query(title="ID of existing credential to upgrade scopes for"),
    ] = None,
) -> LoginResponse:
    handler = _get_provider_oauth_handler(request, provider)

    requested_scopes = scopes.split(",") if scopes else []

    if credential_id:
        requested_scopes = await _prepare_scope_upgrade(
            user_id, provider, credential_id, requested_scopes
        )

    # Generate and store a secure random state token along with the scopes
    state_token, code_challenge = await creds_manager.store.store_state_token(
        user_id, provider, requested_scopes, credential_id=credential_id
    )
    login_url = handler.get_login_url(
        requested_scopes, state_token, code_challenge=code_challenge
    )

    return LoginResponse(login_url=login_url, state_token=state_token)


class CredentialsMetaResponse(BaseModel):
    id: str
    provider: str
    type: CredentialsType
    title: str | None
    scopes: list[str] | None
    username: str | None
    host: str | None = Field(
        default=None,
        description="Host pattern for host-scoped or MCP server URL for MCP credentials",
    )
    is_managed: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_provider(cls, data: Any) -> Any:
        """Fix ``ProviderName.X`` format from Python 3.13 ``str(Enum)`` bug."""
        if isinstance(data, dict):
            prov = data.get("provider", "")
            if isinstance(prov, str) and prov.startswith("ProviderName."):
                member = prov.removeprefix("ProviderName.")
                try:
                    data = {**data, "provider": ProviderName[member].value}
                except KeyError:
                    pass
        return data

    @staticmethod
    def get_host(cred: Credentials) -> str | None:
        """Extract host from credential: HostScoped host or MCP server URL."""
        if isinstance(cred, HostScopedCredentials):
            return cred.host
        if isinstance(cred, OAuth2Credentials) and cred.provider in (
            ProviderName.MCP,
            ProviderName.MCP.value,
            "ProviderName.MCP",
        ):
            return (cred.metadata or {}).get("mcp_server_url")
        return None


def to_meta_response(cred: Credentials) -> CredentialsMetaResponse:
    return CredentialsMetaResponse(
        id=cred.id,
        provider=cred.provider,
        type=cred.type,
        title=cred.title,
        scopes=cred.scopes if isinstance(cred, OAuth2Credentials) else None,
        username=cred.username if isinstance(cred, OAuth2Credentials) else None,
        host=CredentialsMetaResponse.get_host(cred),
        is_managed=cred.is_managed,
    )


@router.post("/{provider}/callback", summary="Exchange OAuth code for tokens")
async def callback(
    provider: Annotated[
        ProviderName, Path(title="The target provider for this OAuth exchange")
    ],
    code: Annotated[str, Body(title="Authorization code acquired by user login")],
    state_token: Annotated[str, Body(title="Anti-CSRF nonce")],
    user_id: Annotated[str, Security(get_user_id)],
    request: Request,
) -> CredentialsMetaResponse:
    logger.debug(f"Received OAuth callback for provider: {provider}")
    handler = _get_provider_oauth_handler(request, provider)

    # Verify the state token
    valid_state = await creds_manager.store.verify_state_token(
        user_id, state_token, provider
    )

    if not valid_state:
        logger.warning(f"Invalid or expired state token for user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired state token",
        )
    try:
        scopes = valid_state.scopes
        logger.debug(f"Retrieved scopes from state token: {scopes}")

        scopes = handler.handle_default_scopes(scopes)

        credentials = await handler.exchange_code_for_tokens(
            code, scopes, valid_state.code_verifier
        )

        logger.debug(f"Received credentials with final scopes: {credentials.scopes}")

        # Linear returns scopes as a single string with spaces, so we need to split them
        # TODO: make a bypass of this part of the OAuth handler
        if len(credentials.scopes) == 1 and " " in credentials.scopes[0]:
            credentials.scopes = credentials.scopes[0].split(" ")

        # Check if the granted scopes are sufficient for the requested scopes
        if not set(scopes).issubset(set(credentials.scopes)):
            # For now, we'll just log the warning and continue
            logger.warning(
                f"Granted scopes {credentials.scopes} for provider {provider.value} "
                f"do not include all requested scopes {scopes}"
            )

    except Exception as e:
        logger.error(
            f"OAuth2 Code->Token exchange failed for provider {provider.value}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth2 callback failed to exchange code for tokens: {str(e)}",
        )

    # TODO: Allow specifying `title` to set on `credentials`
    credentials = await _merge_or_create_credential(
        user_id, provider, credentials, valid_state.credential_id
    )

    logger.debug(
        f"Successfully processed OAuth callback for user {user_id} "
        f"and provider {provider.value}"
    )

    return to_meta_response(credentials)


# Bound the first-time sweep so a slow upstream (e.g. Ayrshare) can't hang
# the credential-list endpoint.  On timeout we still kick off a fire-and-
# forget sweep so provisioning eventually completes; the user just won't
# see the managed cred until the next refresh.
_MANAGED_PROVISION_TIMEOUT_S = 10.0


async def _ensure_managed_credentials_bounded(user_id: str) -> None:
    try:
        await asyncio.wait_for(
            ensure_managed_credentials(user_id, creds_manager.store),
            timeout=_MANAGED_PROVISION_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Managed credential sweep exceeded %.1fs for user=%s; "
            "continuing without it — provisioning will complete in background",
            _MANAGED_PROVISION_TIMEOUT_S,
            user_id,
        )
        asyncio.create_task(ensure_managed_credentials(user_id, creds_manager.store))


@router.get("/credentials", summary="List Credentials")
async def list_credentials(
    user_id: Annotated[str, Security(get_user_id)],
) -> list[CredentialsMetaResponse]:
    # Block on provisioning so managed credentials appear on the first load
    # instead of after a refresh, but with a timeout so a slow upstream
    # can't hang the endpoint.  `_provisioned_users` short-circuits on
    # repeat calls.
    await _ensure_managed_credentials_bounded(user_id)
    credentials = await creds_manager.store.get_all_creds(user_id)

    return [
        to_meta_response(cred) for cred in credentials if not is_sdk_default(cred.id)
    ]


@router.get("/{provider}/credentials")
async def list_credentials_by_provider(
    provider: Annotated[
        ProviderName, Path(title="The provider to list credentials for")
    ],
    user_id: Annotated[str, Security(get_user_id)],
) -> list[CredentialsMetaResponse]:
    await _ensure_managed_credentials_bounded(user_id)
    credentials = await creds_manager.store.get_creds_by_provider(user_id, provider)

    return [
        to_meta_response(cred) for cred in credentials if not is_sdk_default(cred.id)
    ]


@router.get(
    "/{provider}/credentials/{cred_id}", summary="Get Specific Credential By ID"
)
async def get_credential(
    provider: Annotated[
        ProviderName, Path(title="The provider to retrieve credentials for")
    ],
    cred_id: Annotated[str, Path(title="The ID of the credentials to retrieve")],
    user_id: Annotated[str, Security(get_user_id)],
) -> CredentialsMetaResponse:
    if is_sdk_default(cred_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Credentials not found"
        )
    credential = await creds_manager.get(user_id, cred_id)
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Credentials not found"
        )
    if not provider_matches(credential.provider, provider):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Credentials not found"
        )
    return to_meta_response(credential)


class PickerTokenResponse(BaseModel):
    """Short-lived OAuth access token shipped to the browser for rendering a
    provider-hosted picker UI (e.g. Google Drive Picker). Deliberately narrow:
    only the fields the client needs to initialize the picker widget. Issued
    from the user's own stored credential so ownership and scope gating are
    enforced by the credential lookup."""

    access_token: str = Field(
        description="OAuth access token suitable for the picker SDK call."
    )
    access_token_expires_at: int | None = Field(
        default=None,
        description="Unix timestamp at which the access token expires, if known.",
    )


# Allowlist of (provider, scopes) tuples that may mint picker tokens. Only
# Drive-picker-capable scopes qualify so a caller can't use this endpoint to
# extract a GitHub / other-provider OAuth token for unrelated purposes. If a
# future provider integrates a hosted picker that needs a raw access token,
# add its specific picker-relevant scopes here.
_PICKER_TOKEN_ALLOWED_SCOPES: dict[ProviderName, frozenset[str]] = {
    ProviderName.GOOGLE: frozenset(
        [
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive",
        ]
    ),
}


@router.post(
    "/{provider}/credentials/{cred_id}/picker-token",
    summary="Issue a short-lived access token for a provider-hosted picker",
    operation_id="postV1GetPickerToken",
)
async def get_picker_token(
    provider: Annotated[
        ProviderName, Path(title="The provider that owns the credentials")
    ],
    cred_id: Annotated[
        str, Path(title="The ID of the OAuth2 credentials to mint a token from")
    ],
    user_id: Annotated[str, Security(get_user_id)],
) -> PickerTokenResponse:
    """Return the raw access token for an OAuth2 credential so the frontend
    can initialize a provider-hosted picker (e.g. Google Drive Picker).

    `GET /{provider}/credentials/{cred_id}` deliberately strips secrets (see
    `CredentialsMetaResponse` + `TestGetCredentialReturnsMetaOnly` in
    `router_test.py`). That hardening broke the Drive picker, which needs the
    raw access token to call `google.picker.Builder.setOAuthToken(...)`. This
    endpoint carves a narrow, explicit hole: the caller must own the
    credential, it must be OAuth2, and the endpoint returns only the access
    token + its expiry — nothing else about the credential. SDK-default
    credentials are excluded for the same reason as `get_credential`.
    """
    if is_sdk_default(cred_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Credentials not found"
        )

    credential = await creds_manager.get(user_id, cred_id)
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Credentials not found"
        )
    if not provider_matches(credential.provider, provider):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Credentials not found"
        )
    if not isinstance(credential, OAuth2Credentials):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Picker tokens are only available for OAuth2 credentials",
        )
    if not credential.access_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential has no access token; reconnect the account",
        )

    # Gate on provider+scope: only credentials that actually grant access to
    # a provider-hosted picker flow may mint a token through this endpoint.
    # Prevents using this path to extract bearer tokens for unrelated OAuth
    # integrations (e.g. GitHub) that happen to be stored under the same user.
    allowed_scopes = _PICKER_TOKEN_ALLOWED_SCOPES.get(provider)
    if not allowed_scopes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"Picker tokens are not available for provider '{provider.value}'"),
        )
    cred_scopes = set(credential.scopes or [])
    if cred_scopes.isdisjoint(allowed_scopes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Credential does not grant any scope eligible for the picker. "
                "Reconnect with the appropriate scope."
            ),
        )

    return PickerTokenResponse(
        access_token=credential.access_token.get_secret_value(),
        access_token_expires_at=credential.access_token_expires_at,
    )


@router.post("/{provider}/credentials", status_code=201, summary="Create Credentials")
async def create_credentials(
    user_id: Annotated[str, Security(get_user_id)],
    provider: Annotated[
        ProviderName, Path(title="The provider to create credentials for")
    ],
    credentials: Credentials,
) -> CredentialsMetaResponse:
    if is_sdk_default(credentials.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot create credentials with a reserved ID",
        )
    credentials.provider = provider
    try:
        await creds_manager.create(user_id, credentials)
    except Exception:
        logger.exception("Failed to store credentials")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store credentials",
        )
    return to_meta_response(credentials)


class CredentialsDeletionResponse(BaseModel):
    deleted: Literal[True] = True
    revoked: bool | None = Field(
        description="Indicates whether the credentials were also revoked by their "
        "provider. `None`/`null` if not applicable, e.g. when deleting "
        "non-revocable credentials such as API keys."
    )


class CredentialsDeletionNeedsConfirmationResponse(BaseModel):
    deleted: Literal[False] = False
    need_confirmation: Literal[True] = True
    message: str


class AyrshareSSOResponse(BaseModel):
    sso_url: str = Field(..., description="The SSO URL for Ayrshare integration")
    expires_at: datetime = Field(..., description="ISO timestamp when the URL expires")


@router.delete("/{provider}/credentials/{cred_id}")
async def delete_credentials(
    request: Request,
    provider: Annotated[
        ProviderName, Path(title="The provider to delete credentials for")
    ],
    cred_id: Annotated[str, Path(title="The ID of the credentials to delete")],
    user_id: Annotated[str, Security(get_user_id)],
    force: Annotated[
        bool, Query(title="Whether to proceed if any linked webhooks are still in use")
    ] = False,
) -> CredentialsDeletionResponse | CredentialsDeletionNeedsConfirmationResponse:
    if is_sdk_default(cred_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Credentials not found"
        )
    if is_system_credential(cred_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="System-managed credentials cannot be deleted",
        )
    creds = await creds_manager.store.get_creds_by_id(user_id, cred_id)
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Credentials not found"
        )
    if not provider_matches(creds.provider, provider):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credentials not found",
        )
    if creds.is_managed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="AutoGPT-managed credentials cannot be deleted",
        )

    try:
        await remove_all_webhooks_for_credentials(user_id, creds, force)
    except NeedConfirmation as e:
        return CredentialsDeletionNeedsConfirmationResponse(message=str(e))

    await creds_manager.delete(user_id, cred_id)

    tokens_revoked = None
    if isinstance(creds, OAuth2Credentials):
        if provider_matches(provider.value, ProviderName.MCP.value):
            # MCP uses dynamic per-server OAuth — create handler from metadata
            handler = create_mcp_oauth_handler(creds)
        else:
            handler = _get_provider_oauth_handler(request, provider)
        tokens_revoked = await handler.revoke_tokens(creds)

    return CredentialsDeletionResponse(revoked=tokens_revoked)


# ------------------------- WEBHOOK STUFF -------------------------- #


# ⚠️ Note
# No user auth check because this endpoint is for webhook ingress and relies on
# validation by the provider-specific `WebhooksManager`.
@router.post("/{provider}/webhooks/{webhook_id}/ingress")
async def webhook_ingress_generic(
    request: Request,
    provider: Annotated[
        ProviderName, Path(title="Provider where the webhook was registered")
    ],
    webhook_id: Annotated[str, Path(title="Our ID for the webhook")],
):
    logger.debug(f"Received {provider.value} webhook ingress for ID {webhook_id}")
    webhook_manager = get_webhook_manager(provider)
    try:
        webhook = await get_webhook(webhook_id, include_relations=True)
        user_id = webhook.user_id
        credentials = (
            await creds_manager.get(user_id, webhook.credentials_id)
            if webhook.credentials_id
            else None
        )
    except NotFoundError as e:
        logger.warning(f"Webhook payload received for unknown webhook #{webhook_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    logger.debug(f"Webhook #{webhook_id}: {webhook}")
    payload, event_type = await webhook_manager.validate_payload(
        webhook, request, credentials
    )
    logger.debug(
        f"Validated {provider.value} {webhook.webhook_type} {event_type} event "
        f"with payload {payload}"
    )

    webhook_event = WebhookEvent(
        provider=provider,
        webhook_id=webhook_id,
        event_type=event_type,
        payload=payload,
    )
    await publish_webhook_event(webhook_event)
    logger.debug(f"Webhook event published: {webhook_event}")

    if not (webhook.triggered_nodes or webhook.triggered_presets):
        return

    await complete_onboarding_step(user_id, OnboardingStep.TRIGGER_WEBHOOK)

    # Execute all triggers concurrently for better performance
    tasks = []
    tasks.extend(
        _execute_webhook_node_trigger(node, webhook, webhook_id, event_type, payload)
        for node in webhook.triggered_nodes
    )
    tasks.extend(
        _execute_webhook_preset_trigger(
            preset, webhook, webhook_id, event_type, payload
        )
        for preset in webhook.triggered_presets
    )

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@router.post("/webhooks/{webhook_id}/ping")
async def webhook_ping(
    webhook_id: Annotated[str, Path(title="Our ID for the webhook")],
    user_id: Annotated[str, Security(get_user_id)],  # require auth
):
    webhook = await get_webhook(webhook_id)
    webhook_manager = get_webhook_manager(webhook.provider)

    credentials = (
        await creds_manager.get(user_id, webhook.credentials_id)
        if webhook.credentials_id
        else None
    )
    try:
        await webhook_manager.trigger_ping(webhook, credentials)
    except NotImplementedError:
        return False

    if not await wait_for_webhook_event(webhook_id, event_type="ping", timeout=10):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Webhook ping timed out"
        )

    return True


async def _execute_webhook_node_trigger(
    node: NodeModel,
    webhook: WebhookWithRelations,
    webhook_id: str,
    event_type: str,
    payload: dict,
) -> None:
    """Execute a webhook-triggered node."""
    logger.debug(f"Webhook-attached node: {node}")
    if not node.is_triggered_by_event_type(event_type):
        logger.debug(f"Node #{node.id} doesn't trigger on event {event_type}")
        return
    logger.debug(f"Executing graph #{node.graph_id} node #{node.id}")
    try:
        await add_graph_execution(
            user_id=webhook.user_id,
            graph_id=node.graph_id,
            graph_version=node.graph_version,
            nodes_input_masks={node.id: {"payload": payload}},
        )
    except GraphNotInLibraryError as e:
        logger.warning(
            f"Webhook #{webhook_id} execution blocked for "
            f"deleted/archived graph #{node.graph_id} (node #{node.id}): {e}"
        )
        # Clean up orphaned webhook trigger for this graph
        await _cleanup_orphaned_webhook_for_graph(
            node.graph_id, webhook.user_id, webhook_id
        )
    except Exception:
        logger.exception(
            f"Failed to execute graph #{node.graph_id} via webhook #{webhook_id}"
        )
        # Continue processing - webhook should be resilient to individual failures


async def _execute_webhook_preset_trigger(
    preset: LibraryAgentPreset,
    webhook: WebhookWithRelations,
    webhook_id: str,
    event_type: str,
    payload: dict,
) -> None:
    """Execute a webhook-triggered preset."""
    logger.debug(f"Webhook-attached preset: {preset}")
    if not preset.is_active:
        logger.debug(f"Preset #{preset.id} is inactive")
        return

    graph = await get_graph(
        preset.graph_id, preset.graph_version, user_id=webhook.user_id
    )
    if not graph:
        logger.error(
            f"User #{webhook.user_id} has preset #{preset.id} for graph "
            f"#{preset.graph_id} v{preset.graph_version}, "
            "but no access to the graph itself."
        )
        logger.info(f"Automatically deactivating broken preset #{preset.id}")
        await update_preset(preset.user_id, preset.id, is_active=False)
        return
    if not (trigger_node := graph.webhook_input_node):
        # NOTE: this should NEVER happen, but we log and handle it gracefully
        logger.error(
            f"Preset #{preset.id} is triggered by webhook #{webhook.id}, but graph "
            f"#{preset.graph_id} v{preset.graph_version} has no webhook input node"
        )
        await set_preset_webhook(preset.user_id, preset.id, None)
        return
    if not trigger_node.block.is_triggered_by_event_type(preset.inputs, event_type):
        logger.debug(f"Preset #{preset.id} doesn't trigger on event {event_type}")
        return
    logger.debug(f"Executing preset #{preset.id} for webhook #{webhook.id}")

    try:
        await add_graph_execution(
            user_id=webhook.user_id,
            graph_id=preset.graph_id,
            preset_id=preset.id,
            graph_version=preset.graph_version,
            graph_credentials_inputs=preset.credentials,
            nodes_input_masks={trigger_node.id: {**preset.inputs, "payload": payload}},
        )
    except GraphNotInLibraryError as e:
        logger.warning(
            f"Webhook #{webhook_id} execution blocked for "
            f"deleted/archived graph #{preset.graph_id} (preset #{preset.id}): {e}"
        )
        # Clean up orphaned webhook trigger for this graph
        await _cleanup_orphaned_webhook_for_graph(
            preset.graph_id, webhook.user_id, webhook_id
        )
    except Exception:
        logger.exception(
            f"Failed to execute preset #{preset.id} via webhook #{webhook_id}"
        )
        # Continue processing - webhook should be resilient to individual failures


# -------------------- INCREMENTAL AUTH HELPERS -------------------- #


async def _prepare_scope_upgrade(
    user_id: str,
    provider: ProviderName,
    credential_id: str,
    requested_scopes: list[str],
) -> list[str]:
    """Validate an existing credential for scope upgrade and compute scopes.

    For providers without native incremental auth (e.g. GitHub), returns the
    union of existing + requested scopes.  For providers that handle merging
    server-side (e.g. Google with ``include_granted_scopes``), returns the
    requested scopes unchanged.

    Raises HTTPException on validation failure.
    """
    # Platform-owned system credentials must never be upgraded — scope
    # changes here would leak across every user that shares them.
    if is_system_credential(credential_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="System credentials cannot be upgraded",
        )

    existing = await creds_manager.store.get_creds_by_id(user_id, credential_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential to upgrade not found",
        )
    if not isinstance(existing, OAuth2Credentials):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only OAuth2 credentials can be upgraded",
        )
    if not provider_matches(existing.provider, provider.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential provider does not match the requested provider",
        )
    if existing.is_managed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Managed credentials cannot be upgraded",
        )

    # Google handles scope merging via include_granted_scopes; others need
    # the union of existing + new scopes in the login URL.
    if provider != ProviderName.GOOGLE:
        requested_scopes = list(set(requested_scopes) | set(existing.scopes))

    return requested_scopes


async def _merge_or_create_credential(
    user_id: str,
    provider: ProviderName,
    credentials: OAuth2Credentials,
    credential_id: str | None,
) -> OAuth2Credentials:
    """Either upgrade an existing credential or create a new one.

    When *credential_id* is set (explicit upgrade), merges scopes and updates
    the existing credential.  Otherwise, checks for an implicit merge (same
    provider + username) before falling back to creating a new credential.
    """
    if credential_id:
        return await _upgrade_existing_credential(user_id, credential_id, credentials)

    # Implicit merge: check for existing credential with same provider+username.
    # Skip managed/system credentials and require a non-None username on both
    # sides so we never accidentally merge unrelated credentials.
    if credentials.username is None:
        await creds_manager.create(user_id, credentials)
        return credentials

    existing_creds = await creds_manager.store.get_creds_by_provider(user_id, provider)
    matching = next(
        (
            c
            for c in existing_creds
            if isinstance(c, OAuth2Credentials)
            and not c.is_managed
            and not is_system_credential(c.id)
            and c.username is not None
            and c.username == credentials.username
        ),
        None,
    )
    if matching:
        # Only merge into the existing credential when the new token
        # already covers every scope we're about to advertise on it.
        # Without this guard we'd overwrite ``matching.access_token`` with
        # a narrower token while storing a wider ``scopes`` list — the
        # record would claim authorizations the token does not grant, and
        # blocks using the lost scopes would fail with opaque 401/403s
        # until the user hits re-auth.  On a narrowing login, keep the
        # two credentials separate instead.
        if set(credentials.scopes).issuperset(set(matching.scopes)):
            return await _upgrade_existing_credential(user_id, matching.id, credentials)

    await creds_manager.create(user_id, credentials)
    return credentials


async def _upgrade_existing_credential(
    user_id: str,
    existing_cred_id: str,
    new_credentials: OAuth2Credentials,
) -> OAuth2Credentials:
    """Merge scopes from *new_credentials* into an existing credential."""
    # Defense-in-depth: re-check system and provider invariants right before
    # the write.  The login-time check in `_prepare_scope_upgrade` can go stale
    # by the time the callback runs, and the implicit-merge path bypasses
    # login-time validation entirely, so every write-path must enforce these
    # on its own.
    if is_system_credential(existing_cred_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="System credentials cannot be upgraded",
        )
    existing = await creds_manager.store.get_creds_by_id(user_id, existing_cred_id)
    if not existing or not isinstance(existing, OAuth2Credentials):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential to upgrade not found",
        )
    if existing.is_managed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Managed credentials cannot be upgraded",
        )
    if not provider_matches(existing.provider, new_credentials.provider):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential provider does not match the requested provider",
        )

    if (
        existing.username
        and new_credentials.username
        and existing.username != new_credentials.username
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username mismatch: authenticated as a different user",
        )

    # Operate on a copy so the caller's ``new_credentials`` object is not
    # mutated out from under them.  Every caller today immediately discards
    # or replaces its reference, but the implicit-merge path in
    # ``_merge_or_create_credential`` reads ``credentials.scopes`` before
    # calling into us — a future reader after the call would otherwise
    # silently see the overwritten values.
    merged = new_credentials.model_copy(deep=True)
    merged.id = existing.id
    merged.title = existing.title
    merged.scopes = list(set(existing.scopes) | set(new_credentials.scopes))
    merged.metadata = {
        **(existing.metadata or {}),
        **(new_credentials.metadata or {}),
    }
    # Preserve the existing refresh_token and username if the incremental
    # response doesn't carry them.  Providers like Google only return a
    # refresh_token on first authorization — dropping it here would orphan
    # the credential on the next access-token expiry, forcing the user to
    # re-auth from scratch. Username is similarly sticky: if we've already
    # resolved it for this credential, keep it rather than silently
    # blanking it on an incremental upgrade.
    if not merged.refresh_token and existing.refresh_token:
        merged.refresh_token = existing.refresh_token
        merged.refresh_token_expires_at = existing.refresh_token_expires_at
    if not merged.username and existing.username:
        merged.username = existing.username
    await creds_manager.update(user_id, merged)
    return merged


# --------------------------- UTILITIES ---------------------------- #


async def remove_all_webhooks_for_credentials(
    user_id: str, credentials: Credentials, force: bool = False
) -> None:
    """
    Remove and deregister all webhooks that were registered using the given credentials.

    Params:
        user_id: The ID of the user who owns the credentials and webhooks.
        credentials: The credentials for which to remove the associated webhooks.
        force: Whether to proceed if any of the webhooks are still in use.

    Raises:
        NeedConfirmation: If any of the webhooks are still in use and `force` is `False`
    """
    webhooks = await get_all_webhooks_by_creds(
        user_id, credentials.id, include_relations=True
    )
    if any(w.triggered_nodes or w.triggered_presets for w in webhooks) and not force:
        raise NeedConfirmation(
            "Some webhooks linked to these credentials are still in use by an agent"
        )
    for webhook in webhooks:
        # Unlink all nodes & presets
        for node in webhook.triggered_nodes:
            await set_node_webhook(node.id, None)
        for preset in webhook.triggered_presets:
            await set_preset_webhook(user_id, preset.id, None)

        # Prune the webhook
        webhook_manager = get_webhook_manager(ProviderName(credentials.provider))
        success = await webhook_manager.prune_webhook_if_dangling(
            user_id, webhook.id, credentials
        )
        if not success:
            logger.warning(f"Webhook #{webhook.id} failed to prune")


async def _cleanup_orphaned_webhook_for_graph(
    graph_id: str, user_id: str, webhook_id: str
) -> None:
    """
    Clean up orphaned webhook connections for a specific graph when execution fails with GraphNotAccessibleError.
    This happens when an agent is pulled from the Marketplace or deleted
    but webhook triggers still exist.
    """
    try:
        webhook = await get_webhook(webhook_id, include_relations=True)
        if not webhook or webhook.user_id != user_id:
            logger.warning(
                f"Webhook {webhook_id} not found or doesn't belong to user {user_id}"
            )
            return

        nodes_removed = 0
        presets_removed = 0

        # Remove triggered nodes that belong to the deleted graph
        for node in webhook.triggered_nodes:
            if node.graph_id == graph_id:
                try:
                    await set_node_webhook(node.id, None)
                    nodes_removed += 1
                    logger.info(
                        f"Removed orphaned webhook trigger from node {node.id} "
                        f"in deleted/archived graph {graph_id}"
                    )
                except Exception:
                    logger.exception(
                        f"Failed to remove webhook trigger from node {node.id}"
                    )

        # Remove triggered presets that belong to the deleted graph
        for preset in webhook.triggered_presets:
            if preset.graph_id == graph_id:
                try:
                    await set_preset_webhook(user_id, preset.id, None)
                    presets_removed += 1
                    logger.info(
                        f"Removed orphaned webhook trigger from preset {preset.id} "
                        f"for deleted/archived graph {graph_id}"
                    )
                except Exception:
                    logger.exception(
                        f"Failed to remove webhook trigger from preset {preset.id}"
                    )

        if nodes_removed > 0 or presets_removed > 0:
            logger.info(
                f"Cleaned up orphaned webhook #{webhook_id}: "
                f"removed {nodes_removed} nodes and {presets_removed} presets "
                f"for deleted/archived graph #{graph_id}"
            )

            # Check if webhook has any remaining triggers, if not, prune it
            updated_webhook = await get_webhook(webhook_id, include_relations=True)
            if (
                not updated_webhook.triggered_nodes
                and not updated_webhook.triggered_presets
            ):
                try:
                    webhook_manager = get_webhook_manager(
                        ProviderName(webhook.provider)
                    )
                    credentials = (
                        await creds_manager.get(user_id, webhook.credentials_id)
                        if webhook.credentials_id
                        else None
                    )
                    success = await webhook_manager.prune_webhook_if_dangling(
                        user_id, webhook.id, credentials
                    )
                    if success:
                        logger.info(
                            f"Pruned orphaned webhook #{webhook_id} "
                            f"with no remaining triggers"
                        )
                    else:
                        logger.warning(
                            f"Failed to prune orphaned webhook #{webhook_id}"
                        )
                except Exception:
                    logger.exception(f"Failed to prune orphaned webhook #{webhook_id}")

    except Exception:
        logger.exception(
            f"Failed to cleanup orphaned webhook #{webhook_id} for graph #{graph_id}"
        )


def _get_provider_oauth_handler(
    req: Request, provider_name: ProviderName
) -> "BaseOAuthHandler":
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{provider_key}' does not support OAuth",
        )

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
        logger.error(
            f"Attempt to use unconfigured {provider_name.value} OAuth integration"
        )
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={
                "message": f"Integration with provider '{provider_name.value}' is not configured.",
                "hint": "Set client ID and secret in the application's deployment environment",
            },
        )

    handler_class = HANDLERS_BY_NAME[provider_key]
    frontend_base_url = settings.config.frontend_base_url

    if not frontend_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Frontend base URL is not configured",
        )

    return handler_class(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=f"{frontend_base_url}/auth/integrations/oauth_callback",
    )


@router.get("/ayrshare/sso_url")
async def get_ayrshare_sso_url(
    user_id: Annotated[str, Security(get_user_id)],
) -> AyrshareSSOResponse:
    """Generate a JWT SSO URL so the user can link their social accounts.

    The per-user Ayrshare profile key is provisioned and persisted as a
    standard ``is_managed=True`` credential by
    :class:`~backend.integrations.managed_providers.ayrshare.AyrshareManagedProvider`.
    This endpoint only signs a short-lived JWT pointing at the Ayrshare-
    hosted social-linking page; all profile lifecycle logic lives with the
    managed provider.
    """
    if not ayrshare_settings_available():
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ayrshare integration is not configured",
        )

    try:
        client = AyrshareClient()
    except MissingConfigError:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ayrshare integration is not configured",
        )

    # On-demand provisioning: AyrshareManagedProvider opts out of the
    # credentials sweep (profile quota is per-user subscription-bound).  This
    # endpoint is the only trigger that provisions a profile — one Ayrshare
    # profile per user who actually opens the connect flow, not one per
    # every authenticated user.
    provisioned = await ensure_managed_credential(
        user_id, creds_manager.store, AyrshareManagedProvider()
    )
    if not provisioned:
        raise HTTPException(
            status_code=HTTP_502_BAD_GATEWAY,
            detail="Failed to provision Ayrshare profile",
        )

    ayrshare_creds = [
        c
        for c in await creds_manager.store.get_creds_by_provider(user_id, "ayrshare")
        if c.is_managed and isinstance(c, APIKeyCredentials)
    ]
    if not ayrshare_creds:
        logger.error(
            "Ayrshare credential provisioning did not produce a credential "
            "for user %s",
            user_id,
        )
        raise HTTPException(
            status_code=HTTP_502_BAD_GATEWAY,
            detail="Failed to provision Ayrshare profile",
        )
    profile_key_str = ayrshare_creds[0].api_key.get_secret_value()

    private_key = settings.secrets.ayrshare_jwt_key
    # Ayrshare JWT max lifetime is 2880 minutes (48 h).
    max_expiry_minutes = 2880
    try:
        jwt_response = await client.generate_jwt(
            private_key=private_key,
            profile_key=profile_key_str,
            # `allowed_social` is the set of networks the Ayrshare-hosted
            # social-linking page will *offer* the user to connect.  Blocks
            # exist for more platforms than are listed here; the list is
            # deliberately narrower so the rollout can verify each network
            # end-to-end before widening the user-visible surface.  Keep
            # in sync with tested platforms — extend as each is verified
            # against the block + Ayrshare's network-specific quirks.
            allowed_social=[
                SocialPlatform.TWITTER,
                SocialPlatform.LINKEDIN,
                SocialPlatform.INSTAGRAM,
                SocialPlatform.YOUTUBE,
                SocialPlatform.TIKTOK,
            ],
            expires_in=max_expiry_minutes,
            verify=True,
        )
    except Exception as exc:
        logger.error("Error generating Ayrshare JWT for user %s: %s", user_id, exc)
        raise HTTPException(
            status_code=HTTP_502_BAD_GATEWAY, detail="Failed to generate JWT"
        )

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=max_expiry_minutes)
    return AyrshareSSOResponse(sso_url=jwt_response.url, expires_at=expires_at)


# === PROVIDER DISCOVERY ENDPOINTS ===
@router.get("/providers", response_model=List[ProviderMetadata])
async def list_providers() -> List[ProviderMetadata]:
    """
    Get metadata for every available provider.

    Returns both statically defined providers (from ``ProviderName`` enum) and
    dynamically registered providers (from SDK decorators). Each entry includes
    a ``description`` declared via ``ProviderBuilder.with_description(...)`` in
    the provider's ``_config.py``.

    Note: The complete list of provider names is also available as a constant
    in the generated TypeScript client via PROVIDER_NAMES.
    """
    # Ensure all block modules (and therefore every provider's _config.py) are
    # imported before we read from AutoRegistry. Cached on first call.
    try:
        from backend.blocks import load_all_blocks

        load_all_blocks()
    except Exception as e:
        logger.warning(f"Failed to load blocks for provider metadata: {e}")

    all_providers = get_all_provider_names()
    return [
        ProviderMetadata(
            name=name,
            description=get_provider_description(name),
            supported_auth_types=get_supported_auth_types(name),
        )
        for name in all_providers
    ]


@router.get("/providers/system", response_model=List[str])
async def list_system_providers() -> List[str]:
    """
    Get a list of providers that have platform credits (system credentials) available.

    These providers can be used without the user providing their own API keys.
    """
    from backend.integrations.credentials_store import SYSTEM_PROVIDERS

    return list(SYSTEM_PROVIDERS)


@router.get("/providers/names", response_model=ProviderNamesResponse)
async def get_provider_names() -> ProviderNamesResponse:
    """
    Get all provider names in a structured format.

    This endpoint is specifically designed to expose the provider names
    in the OpenAPI schema so that code generators like Orval can create
    appropriate TypeScript constants.
    """
    return ProviderNamesResponse()


@router.get("/providers/constants", response_model=ProviderConstants)
async def get_provider_constants() -> ProviderConstants:
    """
    Get provider names as constants.

    This endpoint returns a model with provider names as constants,
    specifically designed for OpenAPI code generation tools to create
    TypeScript constants.
    """
    return ProviderConstants()


class ProviderEnumResponse(BaseModel):
    """Response containing a provider from the enum."""

    provider: str = Field(
        description="A provider name from the complete list of providers"
    )


@router.get("/providers/enum-example", response_model=ProviderEnumResponse)
async def get_provider_enum_example() -> ProviderEnumResponse:
    """
    Example endpoint that uses the CompleteProviderNames enum.

    This endpoint exists to ensure that the CompleteProviderNames enum is included
    in the OpenAPI schema, which will cause Orval to generate it as a
    TypeScript enum/constant.
    """
    # Return the first provider as an example
    all_providers = get_all_provider_names()
    return ProviderEnumResponse(
        provider=all_providers[0] if all_providers else "openai"
    )
