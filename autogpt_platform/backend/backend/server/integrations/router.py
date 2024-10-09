import logging
from typing import Annotated

from autogpt_libs.supabase_integration_credentials_store.types import (
    APIKeyCredentials,
    Credentials,
    CredentialsType,
    OAuth2Credentials,
)
from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
)
from pydantic import BaseModel, SecretStr

from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.oauth import HANDLERS_BY_NAME, BaseOAuthHandler
from backend.util.settings import Settings

from ..utils import get_user_id

logger = logging.getLogger(__name__)
settings = Settings()
router = APIRouter()
creds_manager = IntegrationCredentialsManager()


class LoginResponse(BaseModel):
    login_url: str
    state_token: str


@router.get("/{provider}/login")
async def login(
    provider: Annotated[str, Path(title="The provider to initiate an OAuth flow for")],
    user_id: Annotated[str, Depends(get_user_id)],
    request: Request,
    scopes: Annotated[
        str, Query(title="Comma-separated list of authorization scopes")
    ] = "",
) -> LoginResponse:
    handler = _get_provider_oauth_handler(request, provider)

    requested_scopes = scopes.split(",") if scopes else []

    # Generate and store a secure random state token along with the scopes
    state_token = await creds_manager.store.store_state_token(
        user_id, provider, requested_scopes
    )

    login_url = handler.get_login_url(requested_scopes, state_token)

    return LoginResponse(login_url=login_url, state_token=state_token)


class CredentialsMetaResponse(BaseModel):
    id: str
    type: CredentialsType
    title: str | None
    scopes: list[str] | None
    username: str | None


@router.post("/{provider}/callback")
async def callback(
    provider: Annotated[str, Path(title="The target provider for this OAuth exchange")],
    code: Annotated[str, Body(title="Authorization code acquired by user login")],
    state_token: Annotated[str, Body(title="Anti-CSRF nonce")],
    user_id: Annotated[str, Depends(get_user_id)],
    request: Request,
) -> CredentialsMetaResponse:
    logger.debug(f"Received OAuth callback for provider: {provider}")
    handler = _get_provider_oauth_handler(request, provider)

    # Verify the state token
    if not await creds_manager.store.verify_state_token(user_id, state_token, provider):
        logger.warning(f"Invalid or expired state token for user {user_id}")
        raise HTTPException(status_code=400, detail="Invalid or expired state token")

    try:
        scopes = await creds_manager.store.get_any_valid_scopes_from_state_token(
            user_id, state_token, provider
        )
        logger.debug(f"Retrieved scopes from state token: {scopes}")

        scopes = handler.handle_default_scopes(scopes)

        credentials = handler.exchange_code_for_tokens(code, scopes)
        logger.debug(f"Received credentials with final scopes: {credentials.scopes}")

        # Check if the granted scopes are sufficient for the requested scopes
        if not set(scopes).issubset(set(credentials.scopes)):
            # For now, we'll just log the warning and continue
            logger.warning(
                f"Granted scopes {credentials.scopes} for {provider}do not include all requested scopes {scopes}"
            )

    except Exception as e:
        logger.error(f"Code->Token exchange failed for provider {provider}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to exchange code for tokens: {str(e)}"
        )

    # TODO: Allow specifying `title` to set on `credentials`
    creds_manager.create(user_id, credentials)

    logger.debug(
        f"Successfully processed OAuth callback for user {user_id} and provider {provider}"
    )
    return CredentialsMetaResponse(
        id=credentials.id,
        type=credentials.type,
        title=credentials.title,
        scopes=credentials.scopes,
        username=credentials.username,
    )


@router.get("/{provider}/credentials")
async def list_credentials(
    provider: Annotated[str, Path(title="The provider to list credentials for")],
    user_id: Annotated[str, Depends(get_user_id)],
) -> list[CredentialsMetaResponse]:
    credentials = creds_manager.store.get_creds_by_provider(user_id, provider)
    return [
        CredentialsMetaResponse(
            id=cred.id,
            type=cred.type,
            title=cred.title,
            scopes=cred.scopes if isinstance(cred, OAuth2Credentials) else None,
            username=cred.username if isinstance(cred, OAuth2Credentials) else None,
        )
        for cred in credentials
    ]


@router.get("/{provider}/credentials/{cred_id}")
async def get_credential(
    provider: Annotated[str, Path(title="The provider to retrieve credentials for")],
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
async def create_api_key_credentials(
    user_id: Annotated[str, Depends(get_user_id)],
    provider: Annotated[str, Path(title="The provider to create credentials for")],
    api_key: Annotated[str, Body(title="The API key to store")],
    title: Annotated[str, Body(title="Optional title for the credentials")],
    expires_at: Annotated[
        int | None, Body(title="Unix timestamp when the key expires")
    ] = None,
) -> APIKeyCredentials:
    new_credentials = APIKeyCredentials(
        provider=provider,
        api_key=SecretStr(api_key),
        title=title,
        expires_at=expires_at,
    )

    try:
        creds_manager.create(user_id, new_credentials)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to store credentials: {str(e)}"
        )
    return new_credentials


@router.delete("/{provider}/credentials/{cred_id}", status_code=204)
async def delete_credential(
    provider: Annotated[str, Path(title="The provider to delete credentials for")],
    cred_id: Annotated[str, Path(title="The ID of the credentials to delete")],
    user_id: Annotated[str, Depends(get_user_id)],
):
    creds = creds_manager.store.get_creds_by_id(user_id, cred_id)
    if not creds:
        raise HTTPException(status_code=404, detail="Credentials not found")
    if creds.provider != provider:
        raise HTTPException(
            status_code=404, detail="Credentials do not match the specified provider"
        )

    creds_manager.delete(user_id, cred_id)
    return Response(status_code=204)


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
    frontend_base_url = settings.config.frontend_base_url or str(req.base_url)
    return handler_class(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=f"{frontend_base_url}/auth/integrations/oauth_callback",
    )
