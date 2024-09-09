import logging
from typing import Annotated, Literal, Optional

from autogpt_libs.supabase_integration_credentials_store import (
    SupabaseIntegrationCredentialsStore,
)
from autogpt_libs.supabase_integration_credentials_store.types import (
    APIKeyCredentials,
    OAuth2Credentials,
)
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel
from supabase import Client

from autogpt_server.integrations.oauth import HANDLERS_BY_NAME, BaseOAuthHandler
from autogpt_server.util.settings import Settings

from .utils import get_supabase, get_user_id

logger = logging.getLogger(__name__)
settings = Settings()
integrations_api_router = APIRouter()


def get_store(supabase: Client = Depends(get_supabase)):
    return SupabaseIntegrationCredentialsStore(supabase)


class LoginResponse(BaseModel):
    login_url: str


@integrations_api_router.get("/{provider}/login")
async def login(
    provider: Annotated[str, Path(title="The provider to initiate an OAuth flow for")],
    user_id: Annotated[str, Depends(get_user_id)],
    request: Request,
    store: Annotated[SupabaseIntegrationCredentialsStore, Depends(get_store)],
    scopes: Annotated[
        str, Query(title="Comma-separated list of authorization scopes")
    ] = "",
) -> LoginResponse:
    handler = _get_provider_oauth_handler(request, provider)

    # Generate and store a secure random state token
    state = await store.store_state_token(user_id, provider)

    requested_scopes = scopes.split(",") if scopes else []
    login_url = handler.get_login_url(requested_scopes, state)

    return LoginResponse(login_url=login_url)


class CredentialsMetaResponse(BaseModel):
    credentials_id: str
    credentials_type: Literal["oauth2", "api_key"]


@integrations_api_router.post("/{provider}/callback")
async def callback(
    provider: Annotated[str, Path(title="The target provider for this OAuth exchange")],
    code: Annotated[str, Body(title="Authorization code acquired by user login")],
    state_token: Annotated[str, Body(title="Anti-CSRF nonce")],
    store: Annotated[SupabaseIntegrationCredentialsStore, Depends(get_store)],
    user_id: Annotated[str, Depends(get_user_id)],
    request: Request,
) -> CredentialsMetaResponse:
    handler = _get_provider_oauth_handler(request, provider)

    # Verify the state token
    if not await store.verify_state_token(user_id, state_token, provider):
        raise HTTPException(status_code=400, detail="Invalid or expired state token")

    try:
        credentials = handler.exchange_code_for_tokens(code)
    except Exception as e:
        logger.warning(f"Code->Token exchange failed for provider {provider}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    store.add_creds(user_id, credentials)
    return CredentialsMetaResponse(
        credentials_id=credentials.id,
        credentials_type=credentials.type,
    )


@integrations_api_router.get(
    "/{provider}/credentials", response_model=list[CredentialsMetaResponse]
)
async def list_credentials(
    provider: Annotated[str, Path(title="The provider to list credentials for")],
    user_id: Annotated[str, Depends(get_user_id)],
    store: Annotated[SupabaseIntegrationCredentialsStore, Depends(get_store)],
) -> list[CredentialsMetaResponse]:
    credentials = store.get_creds_by_provider(user_id, provider)
    return [
        CredentialsMetaResponse(credentials_id=cred.id, credentials_type=cred.type)
        for cred in credentials
    ]


class CredentialsDetailResponse(BaseModel):
    id: str
    provider: str
    title: str
    type: Literal["oauth2", "api_key"]
    access_token: str
    refresh_token: Optional[str]
    access_token_expires_at: Optional[int]
    refresh_token_expires_at: Optional[int]
    scopes: list[str]
    metadata: dict


@integrations_api_router.get(
    "/{provider}/credentials/{cred_id}", response_model=CredentialsDetailResponse
)
async def get_credential(
    provider: Annotated[str, Path(title="The provider to retrieve credentials for")],
    cred_id: Annotated[str, Path(title="The ID of the credentials to retrieve")],
    user_id: Annotated[str, Depends(get_user_id)],
    store: Annotated[SupabaseIntegrationCredentialsStore, Depends(get_store)],
) -> CredentialsDetailResponse:
    credential = store.get_creds_by_id(user_id, cred_id)
    if not credential:
        raise HTTPException(status_code=404, detail="Credentials not found")
    if credential.provider != provider:
        raise HTTPException(
            status_code=400, detail="Credentials do not match the specified provider"
        )

    if isinstance(credential, OAuth2Credentials):
        return CredentialsDetailResponse(
            id=credential.id,
            provider=credential.provider,
            title=credential.title,
            type=credential.type,
            access_token=credential.access_token.get_secret_value(),
            refresh_token=(
                credential.refresh_token.get_secret_value()
                if credential.refresh_token
                else None
            ),
            access_token_expires_at=credential.access_token_expires_at,
            refresh_token_expires_at=credential.refresh_token_expires_at,
            scopes=credential.scopes,
            metadata=credential.metadata,
        )
    elif isinstance(credential, APIKeyCredentials):
        return CredentialsDetailResponse(
            id=credential.id,
            provider=credential.provider,
            title=credential.title,
            type=credential.type,
            access_token=credential.api_key.get_secret_value(),
            refresh_token=None,
            access_token_expires_at=credential.expires_at,
            refresh_token_expires_at=None,
            scopes=[],
            metadata={},
        )
    else:
        raise HTTPException(status_code=500, detail="Unknown credential type")


# -------- UTILITIES --------- #


def _get_provider_oauth_handler(req: Request, provider_name: str) -> BaseOAuthHandler:
    if provider_name not in HANDLERS_BY_NAME:
        raise HTTPException(
            status_code=404, detail=f"Unknown provider '{provider_name}'"
        )

    client_id = getattr(settings.secrets, f"{provider_name}_client_id")
    client_secret = getattr(settings.secrets, f"{provider_name}_client_secret")
    if not (client_id and client_secret):
        raise HTTPException(
            status_code=501,
            detail=f"Integration with provider '{provider_name}' is not configured",
        )

    handler_class = HANDLERS_BY_NAME[provider_name]
    return handler_class(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=str(req.url_for("callback", provider=provider_name)),
    )
