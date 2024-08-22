from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Path, Query, Body
from supabase import Client
from autogpt_libs.supabase_integration_credentials_store import (
    SupabaseIntegrationCredentialsStore,
)

from autogpt_server.integrations.oauth import HANDLERS_BY_NAME, BaseOAuthHandler
from autogpt_server.util.settings import Settings

from .utils import get_user_id, get_supabase

router = APIRouter()

settings = Settings()


def get_store(supabase: Client = Depends(get_supabase)):
    return SupabaseIntegrationCredentialsStore(supabase)


@router.get("/{provider}/login")
async def login(
    provider: Annotated[str, Path(title="The provider to initiate an OAuth flow for")],
    user_id: Annotated[str, Depends(get_user_id)],
    request: Request,
    scopes: Annotated[
        str, Query(title="Comma-separated list of authorization scopes")
    ] = "",
):
    handler = _get_provider_oauth_handler(request, provider)

    state = user_id  # You might want to use a more secure state
    requested_scopes = scopes.split(",") if scopes else []
    login_url = handler.get_login_url(requested_scopes, state)

    return {"login_url": login_url}


@router.post("/{provider}/callback")
async def callback(
    provider: Annotated[str, Path(title="The target provider for this OAuth exchange")],
    code: Annotated[str, Body(title="Authorization code acquired by user login")],
    state: Annotated[str, Body(title="Anti-CSRF nonce")],
    store: Annotated[SupabaseIntegrationCredentialsStore, Depends(get_store)],
    user_id: Annotated[str, Depends(get_user_id)],
    request: Request,
):
    handler = _get_provider_oauth_handler(request, provider)

    # TODO: check state

    try:
        credentials = handler.exchange_code_for_tokens(code)
        store.add_creds(user_id, credentials)
        return {"message": "Authentication successful"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
