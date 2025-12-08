"""
External API endpoints for integrations and credentials.

This module provides endpoints for external applications (like Autopilot) to:
- Initiate OAuth flows with custom callback URLs
- Complete OAuth flows by exchanging authorization codes
- Create API key, user/password, and host-scoped credentials
- List and manage user credentials
"""

import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Union
from urllib.parse import urlparse

from fastapi import APIRouter, Body, HTTPException, Path, Security, status
from prisma.enums import APIKeyPermission
from pydantic import BaseModel, Field, SecretStr

from backend.data.api_key import APIKeyInfo
from backend.data.model import (
    APIKeyCredentials,
    Credentials,
    CredentialsType,
    HostScopedCredentials,
    OAuth2Credentials,
    UserPasswordCredentials,
)
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.oauth import CREDENTIALS_BY_PROVIDER, HANDLERS_BY_NAME
from backend.integrations.providers import ProviderName
from backend.server.external.middleware import require_permission
from backend.server.integrations.models import get_all_provider_names
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.integrations.oauth import BaseOAuthHandler

logger = logging.getLogger(__name__)
settings = Settings()
creds_manager = IntegrationCredentialsManager()

integrations_router = APIRouter(prefix="/integrations", tags=["integrations"])


# ==================== Request/Response Models ==================== #


class OAuthInitiateRequest(BaseModel):
    """Request model for initiating an OAuth flow."""

    callback_url: str = Field(
        ..., description="The external app's callback URL for OAuth redirect"
    )
    scopes: list[str] = Field(
        default_factory=list, description="OAuth scopes to request"
    )
    state_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata to echo back on completion",
    )


class OAuthInitiateResponse(BaseModel):
    """Response model for OAuth initiation."""

    login_url: str = Field(..., description="URL to redirect user for OAuth consent")
    state_token: str = Field(..., description="State token for CSRF protection")
    expires_at: int = Field(
        ..., description="Unix timestamp when the state token expires"
    )


class OAuthCompleteRequest(BaseModel):
    """Request model for completing an OAuth flow."""

    code: str = Field(..., description="Authorization code from OAuth provider")
    state_token: str = Field(..., description="State token from initiate request")


class OAuthCompleteResponse(BaseModel):
    """Response model for OAuth completion."""

    credentials_id: str = Field(..., description="ID of the stored credentials")
    provider: str = Field(..., description="Provider name")
    type: str = Field(..., description="Credential type (oauth2)")
    title: Optional[str] = Field(None, description="Credential title")
    scopes: list[str] = Field(default_factory=list, description="Granted scopes")
    username: Optional[str] = Field(None, description="Username from provider")
    state_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Echoed metadata from initiate request"
    )


class CredentialSummary(BaseModel):
    """Summary of a credential without sensitive data."""

    id: str
    provider: str
    type: CredentialsType
    title: Optional[str] = None
    scopes: Optional[list[str]] = None
    username: Optional[str] = None
    host: Optional[str] = None


class ProviderInfo(BaseModel):
    """Information about an integration provider."""

    name: str
    supports_oauth: bool = False
    supports_api_key: bool = False
    supports_user_password: bool = False
    supports_host_scoped: bool = False
    default_scopes: list[str] = Field(default_factory=list)


# ==================== Credential Creation Models ==================== #


class CreateAPIKeyCredentialRequest(BaseModel):
    """Request model for creating API key credentials."""

    type: Literal["api_key"] = "api_key"
    api_key: str = Field(..., description="The API key")
    title: str = Field(..., description="A name for this credential")
    expires_at: Optional[int] = Field(
        None, description="Unix timestamp when the API key expires"
    )


class CreateUserPasswordCredentialRequest(BaseModel):
    """Request model for creating username/password credentials."""

    type: Literal["user_password"] = "user_password"
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    title: str = Field(..., description="A name for this credential")


class CreateHostScopedCredentialRequest(BaseModel):
    """Request model for creating host-scoped credentials."""

    type: Literal["host_scoped"] = "host_scoped"
    host: str = Field(..., description="Host/domain pattern to match")
    headers: dict[str, str] = Field(..., description="Headers to include in requests")
    title: str = Field(..., description="A name for this credential")


# Union type for credential creation
CreateCredentialRequest = Annotated[
    CreateAPIKeyCredentialRequest
    | CreateUserPasswordCredentialRequest
    | CreateHostScopedCredentialRequest,
    Field(discriminator="type"),
]


class CreateCredentialResponse(BaseModel):
    """Response model for credential creation."""

    id: str
    provider: str
    type: CredentialsType
    title: Optional[str] = None


# ==================== Helper Functions ==================== #


def validate_callback_url(callback_url: str) -> bool:
    """Validate that the callback URL is from an allowed origin."""
    allowed_origins = settings.config.external_oauth_callback_origins

    try:
        parsed = urlparse(callback_url)
        callback_origin = f"{parsed.scheme}://{parsed.netloc}"

        for allowed in allowed_origins:
            # Simple origin matching
            if callback_origin == allowed:
                return True

        # Allow localhost with any port in development (proper hostname check)
        if parsed.hostname == "localhost":
            for allowed in allowed_origins:
                allowed_parsed = urlparse(allowed)
                if allowed_parsed.hostname == "localhost":
                    return True

        return False
    except Exception:
        return False


def _get_oauth_handler_for_external(
    provider_name: str, redirect_uri: str
) -> "BaseOAuthHandler":
    """Get an OAuth handler configured with an external redirect URI."""
    # Ensure blocks are loaded so SDK providers are available
    try:
        from backend.blocks import load_all_blocks

        load_all_blocks()
    except Exception as e:
        logger.warning(f"Failed to load blocks: {e}")

    if provider_name not in HANDLERS_BY_NAME:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{provider_name}' does not support OAuth",
        )

    # Check if this provider has custom OAuth credentials
    oauth_credentials = CREDENTIALS_BY_PROVIDER.get(provider_name)

    if oauth_credentials and not oauth_credentials.use_secrets:
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
        client_id = getattr(settings.secrets, f"{provider_name}_client_id", None)
        client_secret = getattr(
            settings.secrets, f"{provider_name}_client_secret", None
        )

    if not (client_id and client_secret):
        logger.error(f"Attempt to use unconfigured {provider_name} OAuth integration")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={
                "message": f"Integration with provider '{provider_name}' is not configured.",
                "hint": "Set client ID and secret in the application's deployment environment",
            },
        )

    handler_class = HANDLERS_BY_NAME[provider_name]
    return handler_class(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )


# ==================== Endpoints ==================== #


@integrations_router.get("/providers", response_model=list[ProviderInfo])
async def list_providers(
    api_key: APIKeyInfo = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> list[ProviderInfo]:
    """
    List all available integration providers.

    Returns a list of all providers with their supported credential types.
    Most providers support API key credentials, and some also support OAuth.
    """
    # Ensure blocks are loaded
    try:
        from backend.blocks import load_all_blocks

        load_all_blocks()
    except Exception as e:
        logger.warning(f"Failed to load blocks: {e}")

    from backend.sdk.registry import AutoRegistry

    providers = []
    for name in get_all_provider_names():
        supports_oauth = name in HANDLERS_BY_NAME
        handler_class = HANDLERS_BY_NAME.get(name)
        default_scopes = (
            getattr(handler_class, "DEFAULT_SCOPES", []) if handler_class else []
        )

        # Check if provider has specific auth types from SDK registration
        sdk_provider = AutoRegistry.get_provider(name)
        if sdk_provider and sdk_provider.supported_auth_types:
            supports_api_key = "api_key" in sdk_provider.supported_auth_types
            supports_user_password = (
                "user_password" in sdk_provider.supported_auth_types
            )
            supports_host_scoped = "host_scoped" in sdk_provider.supported_auth_types
        else:
            # Fallback for legacy providers
            supports_api_key = True  # All providers can accept API keys
            supports_user_password = name in ("smtp",)
            supports_host_scoped = name == "http"

        providers.append(
            ProviderInfo(
                name=name,
                supports_oauth=supports_oauth,
                supports_api_key=supports_api_key,
                supports_user_password=supports_user_password,
                supports_host_scoped=supports_host_scoped,
                default_scopes=default_scopes,
            )
        )

    return providers


@integrations_router.post(
    "/{provider}/oauth/initiate",
    response_model=OAuthInitiateResponse,
    summary="Initiate OAuth flow",
)
async def initiate_oauth(
    provider: Annotated[str, Path(title="The OAuth provider")],
    request: OAuthInitiateRequest,
    api_key: APIKeyInfo = Security(
        require_permission(APIKeyPermission.MANAGE_INTEGRATIONS)
    ),
) -> OAuthInitiateResponse:
    """
    Initiate an OAuth flow for an external application.

    This endpoint allows external apps to start an OAuth flow with a custom
    callback URL. The callback URL must be from an allowed origin configured
    in the platform settings.

    Returns a login URL to redirect the user to, along with a state token
    for CSRF protection.
    """
    # Validate callback URL
    if not validate_callback_url(request.callback_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Callback URL origin is not allowed. Allowed origins: {settings.config.external_oauth_callback_origins}",
        )

    # Validate provider
    try:
        provider_name = ProviderName(provider)
    except ValueError:
        # Check if it's a dynamically registered provider
        if provider not in HANDLERS_BY_NAME:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider}' not found",
            )
        provider_name = provider

    # Get OAuth handler with external callback URL
    handler = _get_oauth_handler_for_external(
        provider if isinstance(provider_name, str) else provider_name.value,
        request.callback_url,
    )

    # Store state token with external flow metadata
    state_token, code_challenge = await creds_manager.store.store_state_token(
        user_id=api_key.user_id,
        provider=provider if isinstance(provider_name, str) else provider_name.value,
        scopes=request.scopes,
        callback_url=request.callback_url,
        state_metadata=request.state_metadata,
        initiated_by_api_key_id=api_key.id,
    )

    # Build login URL
    login_url = handler.get_login_url(
        request.scopes, state_token, code_challenge=code_challenge
    )

    # Calculate expiration (10 minutes from now)
    from datetime import datetime, timedelta, timezone

    expires_at = int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp())

    return OAuthInitiateResponse(
        login_url=login_url,
        state_token=state_token,
        expires_at=expires_at,
    )


@integrations_router.post(
    "/{provider}/oauth/complete",
    response_model=OAuthCompleteResponse,
    summary="Complete OAuth flow",
)
async def complete_oauth(
    provider: Annotated[str, Path(title="The OAuth provider")],
    request: OAuthCompleteRequest,
    api_key: APIKeyInfo = Security(
        require_permission(APIKeyPermission.MANAGE_INTEGRATIONS)
    ),
) -> OAuthCompleteResponse:
    """
    Complete an OAuth flow by exchanging the authorization code for tokens.

    This endpoint should be called after the user has authorized the application
    and been redirected back to the external app's callback URL with an
    authorization code.
    """
    # Verify state token
    valid_state = await creds_manager.store.verify_state_token(
        api_key.user_id, request.state_token, provider
    )

    if not valid_state:
        logger.warning(f"Invalid or expired state token for provider {provider}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired state token",
        )

    # Verify this is an external flow (callback_url must be set)
    if not valid_state.callback_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="State token was not created for external OAuth flow",
        )

    # Get OAuth handler with the original callback URL
    handler = _get_oauth_handler_for_external(provider, valid_state.callback_url)

    try:
        scopes = valid_state.scopes
        scopes = handler.handle_default_scopes(scopes)

        credentials = await handler.exchange_code_for_tokens(
            request.code, scopes, valid_state.code_verifier
        )

        # Handle Linear's space-separated scopes
        if len(credentials.scopes) == 1 and " " in credentials.scopes[0]:
            credentials.scopes = credentials.scopes[0].split(" ")

        # Check scope mismatch
        if not set(scopes).issubset(set(credentials.scopes)):
            logger.warning(
                f"Granted scopes {credentials.scopes} for provider {provider} "
                f"do not include all requested scopes {scopes}"
            )

    except Exception as e:
        logger.error(f"OAuth2 Code->Token exchange failed for provider {provider}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth2 callback failed to exchange code for tokens: {str(e)}",
        )

    # Store credentials
    await creds_manager.create(api_key.user_id, credentials)

    logger.info(f"Successfully completed external OAuth for provider {provider}")

    return OAuthCompleteResponse(
        credentials_id=credentials.id,
        provider=credentials.provider,
        type=credentials.type,
        title=credentials.title,
        scopes=credentials.scopes,
        username=credentials.username,
        state_metadata=valid_state.state_metadata,
    )


@integrations_router.get("/credentials", response_model=list[CredentialSummary])
async def list_credentials(
    api_key: APIKeyInfo = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> list[CredentialSummary]:
    """
    List all credentials for the authenticated user.

    Returns metadata about each credential without exposing sensitive tokens.
    """
    credentials = await creds_manager.store.get_all_creds(api_key.user_id)
    return [
        CredentialSummary(
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


@integrations_router.get(
    "/{provider}/credentials", response_model=list[CredentialSummary]
)
async def list_credentials_by_provider(
    provider: Annotated[str, Path(title="The provider to list credentials for")],
    api_key: APIKeyInfo = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> list[CredentialSummary]:
    """
    List credentials for a specific provider.
    """
    credentials = await creds_manager.store.get_creds_by_provider(
        api_key.user_id, provider
    )
    return [
        CredentialSummary(
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


@integrations_router.post(
    "/{provider}/credentials",
    response_model=CreateCredentialResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create credentials",
)
async def create_credential(
    provider: Annotated[str, Path(title="The provider to create credentials for")],
    request: Union[
        CreateAPIKeyCredentialRequest,
        CreateUserPasswordCredentialRequest,
        CreateHostScopedCredentialRequest,
    ] = Body(..., discriminator="type"),
    api_key: APIKeyInfo = Security(
        require_permission(APIKeyPermission.MANAGE_INTEGRATIONS)
    ),
) -> CreateCredentialResponse:
    """
    Create non-OAuth credentials for a provider.

    Supports creating:
    - API key credentials (type: "api_key")
    - Username/password credentials (type: "user_password")
    - Host-scoped credentials (type: "host_scoped")

    For OAuth credentials, use the OAuth initiate/complete flow instead.
    """
    # Validate provider exists
    all_providers = get_all_provider_names()
    if provider not in all_providers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{provider}' not found",
        )

    # Create the appropriate credential type
    credentials: Credentials
    if request.type == "api_key":
        credentials = APIKeyCredentials(
            provider=provider,
            api_key=SecretStr(request.api_key),
            title=request.title,
            expires_at=request.expires_at,
        )
    elif request.type == "user_password":
        credentials = UserPasswordCredentials(
            provider=provider,
            username=SecretStr(request.username),
            password=SecretStr(request.password),
            title=request.title,
        )
    elif request.type == "host_scoped":
        # Convert string headers to SecretStr
        secret_headers = {k: SecretStr(v) for k, v in request.headers.items()}
        credentials = HostScopedCredentials(
            provider=provider,
            host=request.host,
            headers=secret_headers,
            title=request.title,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported credential type: {request.type}",
        )

    # Store credentials
    try:
        await creds_manager.create(api_key.user_id, credentials)
    except Exception as e:
        logger.error(f"Failed to store credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store credentials: {str(e)}",
        )

    logger.info(f"Created {request.type} credentials for provider {provider}")

    return CreateCredentialResponse(
        id=credentials.id,
        provider=provider,
        type=credentials.type,
        title=credentials.title,
    )


class DeleteCredentialResponse(BaseModel):
    """Response model for deleting a credential."""

    deleted: bool = Field(..., description="Whether the credential was deleted")
    credentials_id: str = Field(..., description="ID of the deleted credential")


@integrations_router.delete(
    "/{provider}/credentials/{cred_id}",
    response_model=DeleteCredentialResponse,
)
async def delete_credential(
    provider: Annotated[str, Path(title="The provider")],
    cred_id: Annotated[str, Path(title="The credential ID to delete")],
    api_key: APIKeyInfo = Security(
        require_permission(APIKeyPermission.DELETE_INTEGRATIONS)
    ),
) -> DeleteCredentialResponse:
    """
    Delete a credential.

    Note: This does not revoke the tokens with the provider. For full cleanup,
    use the main API's delete endpoint which handles webhook cleanup and
    token revocation.
    """
    creds = await creds_manager.store.get_creds_by_id(api_key.user_id, cred_id)
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Credentials not found"
        )
    if creds.provider != provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credentials do not match the specified provider",
        )

    await creds_manager.delete(api_key.user_id, cred_id)

    return DeleteCredentialResponse(deleted=True, credentials_id=cred_id)
