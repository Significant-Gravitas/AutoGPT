import logging
import typing

import autogpt_libs.supabase_integration_credentials_store
import fastapi
import pydantic
import supabase

import autogpt_server.integrations.oauth
import autogpt_server.util.settings

import autogpt_server.server.utils

logger = logging.getLogger(__name__)
settings = autogpt_server.util.settings.Settings()
integrations_api_router = fastapi.APIRouter()


def get_store(
    supabase: supabase.Client = fastapi.Depends(
        autogpt_server.server.utils.get_supabase
    ),
):
    return autogpt_libs.supabase_integration_credentials_store.SupabaseIntegrationCredentialsStore(
        supabase
    )


class LoginResponse(pydantic.BaseModel):
    login_url: str


@integrations_api_router.get("/{provider}/login")
async def login(
    provider: typing.Annotated[
        str, fastapi.Path(title="The provider to initiate an OAuth flow for")
    ],
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
    request: fastapi.Request,
    store: typing.Annotated[
        autogpt_libs.supabase_integration_credentials_store.SupabaseIntegrationCredentialsStore,
        fastapi.Depends(get_store),
    ],
    scopes: typing.Annotated[
        str, fastapi.Query(title="Comma-separated list of authorization scopes")
    ] = "",
) -> LoginResponse:
    handler = _get_provider_oauth_handler(request, provider)

    # Generate and store a secure random state token
    state = await store.store_state_token(user_id, provider)

    requested_scopes = scopes.split(",") if scopes else []
    login_url = handler.get_login_url(requested_scopes, state)

    return LoginResponse(login_url=login_url)


class CredentialsMetaResponse(pydantic.BaseModel):
    credentials_id: str
    credentials_type: typing.Literal["oauth2", "api_key"]


@integrations_api_router.post("/{provider}/callback")
async def callback(
    provider: typing.Annotated[
        str, fastapi.Path(title="The target provider for this OAuth exchange")
    ],
    code: typing.Annotated[
        str, fastapi.Body(title="Authorization code acquired by user login")
    ],
    state_token: typing.Annotated[str, fastapi.Body(title="Anti-CSRF nonce")],
    store: typing.Annotated[
        autogpt_libs.supabase_integration_credentials_store.SupabaseIntegrationCredentialsStore,
        fastapi.Depends(get_store),
    ],
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_server.server.utils.get_user_id)
    ],
    request: fastapi.Request,
) -> CredentialsMetaResponse:
    handler = _get_provider_oauth_handler(request, provider)

    # Verify the state token
    if not await store.verify_state_token(user_id, state_token, provider):
        raise fastapi.HTTPException(
            status_code=400, detail="Invalid or expired state token"
        )

    try:
        credentials = handler.exchange_code_for_tokens(code)
    except Exception as e:
        logger.warning(f"Code->Token exchange failed for provider {provider}: {e}")
        raise fastapi.HTTPException(status_code=400, detail=str(e))

    store.add_creds(user_id, credentials)
    return CredentialsMetaResponse(
        credentials_id=credentials.id,
        credentials_type=credentials.type,
    )


# -------- UTILITIES --------- #


def _get_provider_oauth_handler(
    req: fastapi.Request, provider_name: str
) -> autogpt_server.integrations.oauth.BaseOAuthHandler:
    if provider_name not in autogpt_server.integrations.oauth.HANDLERS_BY_NAME:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Unknown provider '{provider_name}'"
        )

    client_id = getattr(settings.secrets, f"{provider_name}_client_id")
    client_secret = getattr(settings.secrets, f"{provider_name}_client_secret")
    if not (client_id and client_secret):
        raise fastapi.HTTPException(
            status_code=501,
            detail=f"Integration with provider '{provider_name}' is not configured",
        )

    handler_class = autogpt_server.integrations.oauth.HANDLERS_BY_NAME[provider_name]
    return handler_class(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=str(req.url_for("callback", provider=provider_name)),
    )
