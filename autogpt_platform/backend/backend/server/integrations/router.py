import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Annotated, Awaitable, List, Literal

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
from pydantic import BaseModel, Field, SecretStr
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_502_BAD_GATEWAY

from backend.data.graph import get_graph, set_node_webhook
from backend.data.integrations import (
    WebhookEvent,
    get_all_webhooks_by_creds,
    get_webhook,
    publish_webhook_event,
    wait_for_webhook_event,
)
from backend.data.model import (
    Credentials,
    CredentialsType,
    HostScopedCredentials,
    OAuth2Credentials,
    UserIntegrations,
)
from backend.data.onboarding import complete_webhook_trigger_step
from backend.data.user import get_user_integrations
from backend.executor.utils import add_graph_execution
from backend.integrations.ayrshare import AyrshareClient, SocialPlatform
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.oauth import CREDENTIALS_BY_PROVIDER, HANDLERS_BY_NAME
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks import get_webhook_manager
from backend.server.integrations.models import (
    ProviderConstants,
    ProviderNamesResponse,
    get_all_provider_names,
)
from backend.server.v2.library.db import set_preset_webhook, update_preset
from backend.util.exceptions import MissingConfigError, NeedConfirmation, NotFoundError
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.integrations.oauth import BaseOAuthHandler

logger = logging.getLogger(__name__)
settings = Settings()
router = APIRouter()

creds_manager = IntegrationCredentialsManager()


class LoginResponse(BaseModel):
    login_url: str
    state_token: str


@router.get("/{provider}/login")
async def login(
    provider: Annotated[
        ProviderName, Path(title="The provider to initiate an OAuth flow for")
    ],
    user_id: Annotated[str, Security(get_user_id)],
    request: Request,
    scopes: Annotated[
        str, Query(title="Comma-separated list of authorization scopes")
    ] = "",
) -> LoginResponse:
    handler = _get_provider_oauth_handler(request, provider)

    requested_scopes = scopes.split(",") if scopes else []

    # Generate and store a secure random state token along with the scopes
    state_token, code_challenge = await creds_manager.store.store_state_token(
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
    host: str | None = Field(
        default=None, description="Host pattern for host-scoped credentials"
    )


@router.post("/{provider}/callback")
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
    await creds_manager.create(user_id, credentials)

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
        host=(
            credentials.host if isinstance(credentials, HostScopedCredentials) else None
        ),
    )


@router.get("/credentials", summary="List Credentials")
async def list_credentials(
    user_id: Annotated[str, Security(get_user_id)],
) -> list[CredentialsMetaResponse]:
    credentials = await creds_manager.store.get_all_creds(user_id)
    return [
        CredentialsMetaResponse(
            id=cred.id,
            provider=cred.provider,
            type=cred.type,
            title=cred.title,
            scopes=cred.scopes if isinstance(cred, OAuth2Credentials) else None,
            username=cred.username if isinstance(cred, OAuth2Credentials) else None,
            host=cred.host if isinstance(cred, HostScopedCredentials) else None,
        )
        for cred in credentials
    ]


@router.get("/{provider}/credentials")
async def list_credentials_by_provider(
    provider: Annotated[
        ProviderName, Path(title="The provider to list credentials for")
    ],
    user_id: Annotated[str, Security(get_user_id)],
) -> list[CredentialsMetaResponse]:
    credentials = await creds_manager.store.get_creds_by_provider(user_id, provider)
    return [
        CredentialsMetaResponse(
            id=cred.id,
            provider=cred.provider,
            type=cred.type,
            title=cred.title,
            scopes=cred.scopes if isinstance(cred, OAuth2Credentials) else None,
            username=cred.username if isinstance(cred, OAuth2Credentials) else None,
            host=cred.host if isinstance(cred, HostScopedCredentials) else None,
        )
        for cred in credentials
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
) -> Credentials:
    credential = await creds_manager.get(user_id, cred_id)
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Credentials not found"
        )
    if credential.provider != provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credentials do not match the specified provider",
        )
    return credential


@router.post("/{provider}/credentials", status_code=201, summary="Create Credentials")
async def create_credentials(
    user_id: Annotated[str, Security(get_user_id)],
    provider: Annotated[
        ProviderName, Path(title="The provider to create credentials for")
    ],
    credentials: Credentials,
) -> Credentials:
    credentials.provider = provider
    try:
        await creds_manager.create(user_id, credentials)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store credentials: {str(e)}",
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
    creds = await creds_manager.store.get_creds_by_id(user_id, cred_id)
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Credentials not found"
        )
    if creds.provider != provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credentials do not match the specified provider",
        )

    try:
        await remove_all_webhooks_for_credentials(user_id, creds, force)
    except NeedConfirmation as e:
        return CredentialsDeletionNeedsConfirmationResponse(message=str(e))

    await creds_manager.delete(user_id, cred_id)

    tokens_revoked = None
    if isinstance(creds, OAuth2Credentials):
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

    executions: list[Awaitable] = []
    await complete_webhook_trigger_step(user_id)

    for node in webhook.triggered_nodes:
        logger.debug(f"Webhook-attached node: {node}")
        if not node.is_triggered_by_event_type(event_type):
            logger.debug(f"Node #{node.id} doesn't trigger on event {event_type}")
            continue
        logger.debug(f"Executing graph #{node.graph_id} node #{node.id}")
        executions.append(
            add_graph_execution(
                user_id=webhook.user_id,
                graph_id=node.graph_id,
                graph_version=node.graph_version,
                nodes_input_masks={node.id: {"payload": payload}},
            )
        )
    for preset in webhook.triggered_presets:
        logger.debug(f"Webhook-attached preset: {preset}")
        if not preset.is_active:
            logger.debug(f"Preset #{preset.id} is inactive")
            continue

        graph = await get_graph(preset.graph_id, preset.graph_version, webhook.user_id)
        if not graph:
            logger.error(
                f"User #{webhook.user_id} has preset #{preset.id} for graph "
                f"#{preset.graph_id} v{preset.graph_version}, "
                "but no access to the graph itself."
            )
            logger.info(f"Automatically deactivating broken preset #{preset.id}")
            await update_preset(preset.user_id, preset.id, is_active=False)
            continue
        if not (trigger_node := graph.webhook_input_node):
            # NOTE: this should NEVER happen, but we log and handle it gracefully
            logger.error(
                f"Preset #{preset.id} is triggered by webhook #{webhook.id}, but graph "
                f"#{preset.graph_id} v{preset.graph_version} has no webhook input node"
            )
            await set_preset_webhook(preset.user_id, preset.id, None)
            continue
        if not trigger_node.block.is_triggered_by_event_type(preset.inputs, event_type):
            logger.debug(f"Preset #{preset.id} doesn't trigger on event {event_type}")
            continue
        logger.debug(f"Executing preset #{preset.id} for webhook #{webhook.id}")

        executions.append(
            add_graph_execution(
                user_id=webhook.user_id,
                graph_id=preset.graph_id,
                preset_id=preset.id,
                graph_version=preset.graph_version,
                graph_credentials_inputs=preset.credentials,
                nodes_input_masks={
                    trigger_node.id: {**preset.inputs, "payload": payload}
                },
            )
        )
    asyncio.gather(*executions)


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
    """
    Generate an SSO URL for Ayrshare social media integration.

    Returns:
        dict: Contains the SSO URL for Ayrshare integration
    """
    try:
        client = AyrshareClient()
    except MissingConfigError:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ayrshare integration is not configured",
        )

    # Ayrshare profile key is stored in the credentials store
    # It is generated when creating a new profile, if there is no profile key,
    # we create a new profile and store the profile key in the credentials store

    user_integrations: UserIntegrations = await get_user_integrations(user_id)
    profile_key = user_integrations.managed_credentials.ayrshare_profile_key

    if not profile_key:
        logger.debug(f"Creating new Ayrshare profile for user {user_id}")
        try:
            profile = await client.create_profile(
                title=f"User {user_id}", messaging_active=True
            )
            profile_key = profile.profileKey
            await creds_manager.store.set_ayrshare_profile_key(user_id, profile_key)
        except Exception as e:
            logger.error(f"Error creating Ayrshare profile for user {user_id}: {e}")
            raise HTTPException(
                status_code=HTTP_502_BAD_GATEWAY,
                detail="Failed to create Ayrshare profile",
            )
    else:
        logger.debug(f"Using existing Ayrshare profile for user {user_id}")

    profile_key_str = (
        profile_key.get_secret_value()
        if isinstance(profile_key, SecretStr)
        else str(profile_key)
    )

    private_key = settings.secrets.ayrshare_jwt_key
    # Ayrshare JWT expiry is 2880 minutes (48 hours)
    max_expiry_minutes = 2880
    try:
        logger.debug(f"Generating Ayrshare JWT for user {user_id}")
        jwt_response = await client.generate_jwt(
            private_key=private_key,
            profile_key=profile_key_str,
            allowed_social=[
                # NOTE: We are enabling platforms one at a time
                # to speed up the development process
                # SocialPlatform.FACEBOOK,
                SocialPlatform.TWITTER,
                SocialPlatform.LINKEDIN,
                SocialPlatform.INSTAGRAM,
                SocialPlatform.YOUTUBE,
                # SocialPlatform.REDDIT,
                # SocialPlatform.TELEGRAM,
                # SocialPlatform.GOOGLE_MY_BUSINESS,
                # SocialPlatform.PINTEREST,
                SocialPlatform.TIKTOK,
                # SocialPlatform.BLUESKY,
                # SocialPlatform.SNAPCHAT,
                # SocialPlatform.THREADS,
            ],
            expires_in=max_expiry_minutes,
            verify=True,
        )
    except Exception as e:
        logger.error(f"Error generating Ayrshare JWT for user {user_id}: {e}")
        raise HTTPException(
            status_code=HTTP_502_BAD_GATEWAY, detail="Failed to generate JWT"
        )

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=max_expiry_minutes)
    return AyrshareSSOResponse(sso_url=jwt_response.url, expires_at=expires_at)


# === PROVIDER DISCOVERY ENDPOINTS ===
@router.get("/providers", response_model=List[str])
async def list_providers() -> List[str]:
    """
    Get a list of all available provider names.

    Returns both statically defined providers (from ProviderName enum)
    and dynamically registered providers (from SDK decorators).

    Note: The complete list of provider names is also available as a constant
    in the generated TypeScript client via PROVIDER_NAMES.
    """
    # Get all providers at runtime
    all_providers = get_all_provider_names()
    return all_providers


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
