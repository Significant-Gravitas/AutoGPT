import logging
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from backend.data.graph import set_node_webhook
from backend.data.integrations import (
    WebhookEvent,
    get_all_webhooks_by_creds,
    get_webhook,
    publish_webhook_event,
    wait_for_webhook_event,
)
from backend.data.model import Credentials, CredentialsType, OAuth2Credentials
from backend.executor.manager import ExecutionManager
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.oauth import HANDLERS_BY_NAME
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks import WEBHOOK_MANAGERS_BY_NAME
from backend.util.exceptions import NeedConfirmation
from backend.util.service import get_service_client
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.integrations.oauth import BaseOAuthHandler

from ..utils import get_user_id

logger = logging.getLogger(__name__)
settings = Settings()
router = APIRouter()

creds_manager = IntegrationCredentialsManager()


class LoginResponse(BaseModel):
    login_url: str
    state_token: str


@router.get("/{provider}/login")
def login(
    provider: Annotated[
        ProviderName, Path(title="The provider to initiate an OAuth flow for")
    ],
    user_id: Annotated[str, Depends(get_user_id)],
    request: Request,
    scopes: Annotated[
        str, Query(title="Comma-separated list of authorization scopes")
    ] = "",
) -> LoginResponse:
    handler = _get_provider_oauth_handler(request, provider)

    requested_scopes = scopes.split(",") if scopes else []

    # Generate and store a secure random state token along with the scopes
    state_token, code_challenge = creds_manager.store.store_state_token(
        user_id, provider, requested_scopes
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


@router.post("/{provider}/callback")
def callback(
    provider: Annotated[
        ProviderName, Path(title="The target provider for this OAuth exchange")
    ],
    code: Annotated[str, Body(title="Authorization code acquired by user login")],
    state_token: Annotated[str, Body(title="Anti-CSRF nonce")],
    user_id: Annotated[str, Depends(get_user_id)],
    request: Request,
) -> CredentialsMetaResponse:
    logger.debug(f"Received OAuth callback for provider: {provider}")
    handler = _get_provider_oauth_handler(request, provider)

    # Verify the state token
    valid_state = creds_manager.store.verify_state_token(user_id, state_token, provider)

    if not valid_state:
        logger.warning(f"Invalid or expired state token for user {user_id}")
        raise HTTPException(status_code=400, detail="Invalid or expired state token")
    try:
        scopes = valid_state.scopes
        logger.debug(f"Retrieved scopes from state token: {scopes}")

        scopes = handler.handle_default_scopes(scopes)

        credentials = handler.exchange_code_for_tokens(
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
        logger.error(f"Code->Token exchange failed for provider {provider.value}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to exchange code for tokens: {str(e)}"
        )

    # TODO: Allow specifying `title` to set on `credentials`
    creds_manager.create(user_id, credentials)

    logger.debug(
        f"Successfully processed OAuth callback for user {user_id} "
        f"and provider {provider.value}"
    )
    return CredentialsMetaResponse(
        id=credentials.id,
        provider=credentials.provider,
        type=credentials.type,
        title=credentials.title,
        scopes=credentials.scopes,
        username=credentials.username,
    )


@router.get("/credentials")
def list_credentials(
    user_id: Annotated[str, Depends(get_user_id)],
) -> list[CredentialsMetaResponse]:
    credentials = creds_manager.store.get_all_creds(user_id)
    return [
        CredentialsMetaResponse(
            id=cred.id,
            provider=cred.provider,
            type=cred.type,
            title=cred.title,
            scopes=cred.scopes if isinstance(cred, OAuth2Credentials) else None,
            username=cred.username if isinstance(cred, OAuth2Credentials) else None,
        )
        for cred in credentials
    ]


@router.get("/{provider}/credentials")
def list_credentials_by_provider(
    provider: Annotated[
        ProviderName, Path(title="The provider to list credentials for")
    ],
    user_id: Annotated[str, Depends(get_user_id)],
) -> list[CredentialsMetaResponse]:
    credentials = creds_manager.store.get_creds_by_provider(user_id, provider)
    return [
        CredentialsMetaResponse(
            id=cred.id,
            provider=cred.provider,
            type=cred.type,
            title=cred.title,
            scopes=cred.scopes if isinstance(cred, OAuth2Credentials) else None,
            username=cred.username if isinstance(cred, OAuth2Credentials) else None,
        )
        for cred in credentials
    ]


@router.get("/{provider}/credentials/{cred_id}")
def get_credential(
    provider: Annotated[
        ProviderName, Path(title="The provider to retrieve credentials for")
    ],
    cred_id: Annotated[str, Path(title="The ID of the credentials to retrieve")],
    user_id: Annotated[str, Depends(get_user_id)],
) -> Credentials:
    credential = creds_manager.get(user_id, cred_id)
    if not credential:
        raise HTTPException(status_code=404, detail="Credentials not found")
    if credential.provider != provider:
        raise HTTPException(
            status_code=404, detail="Credentials do not match the specified provider"
        )
    return credential


@router.post("/{provider}/credentials", status_code=201)
def create_credentials(
    user_id: Annotated[str, Depends(get_user_id)],
    provider: Annotated[
        ProviderName, Path(title="The provider to create credentials for")
    ],
    credentials: Credentials,
) -> Credentials:
    credentials.provider = provider
    try:
        creds_manager.create(user_id, credentials)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to store credentials: {str(e)}"
        )
    return credentials


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


@router.delete("/{provider}/credentials/{cred_id}")
async def delete_credentials(
    request: Request,
    provider: Annotated[
        ProviderName, Path(title="The provider to delete credentials for")
    ],
    cred_id: Annotated[str, Path(title="The ID of the credentials to delete")],
    user_id: Annotated[str, Depends(get_user_id)],
    force: Annotated[
        bool, Query(title="Whether to proceed if any linked webhooks are still in use")
    ] = False,
) -> CredentialsDeletionResponse | CredentialsDeletionNeedsConfirmationResponse:
    creds = creds_manager.store.get_creds_by_id(user_id, cred_id)
    if not creds:
        raise HTTPException(status_code=404, detail="Credentials not found")
    if creds.provider != provider:
        raise HTTPException(
            status_code=404, detail="Credentials do not match the specified provider"
        )

    try:
        await remove_all_webhooks_for_credentials(creds, force)
    except NeedConfirmation as e:
        return CredentialsDeletionNeedsConfirmationResponse(message=str(e))

    creds_manager.delete(user_id, cred_id)

    tokens_revoked = None
    if isinstance(creds, OAuth2Credentials):
        handler = _get_provider_oauth_handler(request, provider)
        tokens_revoked = handler.revoke_tokens(creds)

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
    webhook_manager = WEBHOOK_MANAGERS_BY_NAME[provider]()
    webhook = await get_webhook(webhook_id)
    logger.debug(f"Webhook #{webhook_id}: {webhook}")
    payload, event_type = await webhook_manager.validate_payload(webhook, request)
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

    if not webhook.attached_nodes:
        return

    executor = get_service_client(ExecutionManager)
    for node in webhook.attached_nodes:
        logger.debug(f"Webhook-attached node: {node}")
        if not node.is_triggered_by_event_type(event_type):
            logger.debug(f"Node #{node.id} doesn't trigger on event {event_type}")
            continue
        logger.debug(f"Executing graph #{node.graph_id} node #{node.id}")
        executor.add_execution(
            graph_id=node.graph_id,
            graph_version=node.graph_version,
            data={f"webhook_{webhook_id}_payload": payload},
            user_id=webhook.user_id,
        )


@router.post("/webhooks/{webhook_id}/ping")
async def webhook_ping(
    webhook_id: Annotated[str, Path(title="Our ID for the webhook")],
    user_id: Annotated[str, Depends(get_user_id)],  # require auth
):
    webhook = await get_webhook(webhook_id)
    webhook_manager = WEBHOOK_MANAGERS_BY_NAME[webhook.provider]()

    credentials = (
        creds_manager.get(user_id, webhook.credentials_id)
        if webhook.credentials_id
        else None
    )
    try:
        await webhook_manager.trigger_ping(webhook, credentials)
    except NotImplementedError:
        return False

    if not await wait_for_webhook_event(webhook_id, event_type="ping", timeout=10):
        raise HTTPException(status_code=504, detail="Webhook ping timed out")

    return True


# --------------------------- UTILITIES ---------------------------- #


async def remove_all_webhooks_for_credentials(
    credentials: Credentials, force: bool = False
) -> None:
    """
    Remove and deregister all webhooks that were registered using the given credentials.

    Params:
        credentials: The credentials for which to remove the associated webhooks.
        force: Whether to proceed if any of the webhooks are still in use.

    Raises:
        NeedConfirmation: If any of the webhooks are still in use and `force` is `False`
    """
    webhooks = await get_all_webhooks_by_creds(credentials.id)
    if credentials.provider not in WEBHOOK_MANAGERS_BY_NAME:
        if webhooks:
            logger.error(
                f"Credentials #{credentials.id} for provider {credentials.provider} "
                f"are attached to {len(webhooks)} webhooks, "
                f"but there is no available WebhooksHandler for {credentials.provider}"
            )
        return
    if any(w.attached_nodes for w in webhooks) and not force:
        raise NeedConfirmation(
            "Some webhooks linked to these credentials are still in use by an agent"
        )
    for webhook in webhooks:
        # Unlink all nodes
        for node in webhook.attached_nodes or []:
            await set_node_webhook(node.id, None)

        # Prune the webhook
        webhook_manager = WEBHOOK_MANAGERS_BY_NAME[credentials.provider]()
        success = await webhook_manager.prune_webhook_if_dangling(
            webhook.id, credentials
        )
        if not success:
            logger.warning(f"Webhook #{webhook.id} failed to prune")


def _get_provider_oauth_handler(
    req: Request, provider_name: ProviderName
) -> "BaseOAuthHandler":
    if provider_name not in HANDLERS_BY_NAME:
        raise HTTPException(
            status_code=404,
            detail=f"Provider '{provider_name.value}' does not support OAuth",
        )

    client_id = getattr(settings.secrets, f"{provider_name.value}_client_id")
    client_secret = getattr(settings.secrets, f"{provider_name.value}_client_secret")
    if not (client_id and client_secret):
        raise HTTPException(
            status_code=501,
            detail=(
                f"Integration with provider '{provider_name.value}' is not configured"
            ),
        )

    handler_class = HANDLERS_BY_NAME[provider_name]
    frontend_base_url = (
        settings.config.frontend_base_url
        or settings.config.platform_base_url
        or str(req.base_url)
    )
    return handler_class(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=f"{frontend_base_url}/auth/integrations/oauth_callback",
    )
