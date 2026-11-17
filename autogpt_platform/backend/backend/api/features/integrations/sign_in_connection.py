"""Connections that come with a Google or Microsoft sign-in.

Login can ask Google or Microsoft for the data scopes too (the frontend's
``lib/auth/connect-at-sign-in.ts``). The frontend's server then hands this
endpoint the tokens from that sign-in, so the person starts with the
connection the Connect flow would have made, without a second consent screen.

Only the frontend's server can call it: its service token names the user who
just signed in. A sign-in never adds a second connection for an account that
already has one: it widens the existing one when the new token covers it, and
otherwise leaves it alone.
"""

import logging
import time
from typing import Annotated, Literal

import jwt
from autogpt_libs.auth.service import frontend_service_claims
from fastapi import APIRouter, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.credentials_store import is_system_credential
from backend.integrations.oauth.google import GoogleOAuthHandler
from backend.integrations.oauth.microsoft_tokens import (
    credentials_from_tokens,
    microsoft_app,
)
from backend.integrations.oauth.sign_in import (
    SignInClientMismatch,
    SignInConnectionError,
)
from backend.integrations.providers import ProviderName

from .router import (
    _get_provider_oauth_handler,
    _same_account,
    _upgrade_existing_credential,
    creds_manager,
)

logger = logging.getLogger(__name__)

SIGN_IN_CONNECTION_SCOPE = "integrations:sign-in-connection"

sign_in_connection_router = APIRouter()
_bearer = HTTPBearer(auto_error=False)


class SignInConnectionRequest(BaseModel):
    provider: Literal["google", "microsoft"]
    access_token: SecretStr
    refresh_token: SecretStr | None = None
    # Unix seconds.
    expires_at: int | None = None
    # What the provider granted, space-separated, as its token response says.
    scope: str = ""
    # Microsoft names the account in it; Google's isn't needed.
    id_token: SecretStr | None = None


SignInConnectionResult = Literal["created", "updated", "kept", "skipped"]


class SignInConnectionResponse(BaseModel):
    result: SignInConnectionResult


async def signed_in_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)],
) -> str:
    """The user the frontend's service token says just signed in."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing",
        )
    claims = await frontend_service_claims(
        credentials.credentials, SIGN_IN_CONNECTION_SCOPE
    )
    user_id = claims.get("user_id") if claims else None
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not a sign-in connection token",
        )
    return user_id


@sign_in_connection_router.post(
    "/sign-in-connection",
    summary="Keep the connection a Google or Microsoft sign-in granted",
    operation_id="connectFromSignIn",
    responses={
        401: {"description": "Not the frontend's sign-in connection token"},
        409: {"description": "Login and connections use different OAuth clients"},
        422: {"description": "The provider didn't accept the sign-in's tokens"},
        501: {"description": "The provider's connections aren't configured"},
    },
)
async def connect_from_sign_in(
    request: SignInConnectionRequest,
    user_id: Annotated[str, Security(signed_in_user)],
) -> SignInConnectionResponse:
    try:
        credentials = await _credentials_from_sign_in(request)
    except SignInClientMismatch as error:
        logger.warning("Sign-in connection refused: %s", error)
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(error)) from None
    except SignInConnectionError as error:
        logger.warning("Sign-in connection refused: %s", error)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from None
    result = await keep_sign_in_connection(user_id, credentials)
    logger.info("Sign-in connection for %s: %s", request.provider, result)
    return SignInConnectionResponse(result=result)


async def _credentials_from_sign_in(
    request: SignInConnectionRequest,
) -> OAuth2Credentials:
    access_token = request.access_token.get_secret_value()
    refresh_token = (
        request.refresh_token.get_secret_value() if request.refresh_token else None
    )
    if request.provider == "google":
        handler = _get_provider_oauth_handler(ProviderName.GOOGLE)
        assert isinstance(handler, GoogleOAuthHandler)
        return await handler.credentials_from_sign_in(
            access_token, refresh_token, request.expires_at
        )
    return await _microsoft_credentials(request, access_token, refresh_token)


async def _microsoft_credentials(
    request: SignInConnectionRequest, access_token: str, refresh_token: str | None
) -> OAuth2Credentials:
    payload = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "scope": request.scope,
        "id_token": request.id_token.get_secret_value() if request.id_token else None,
        "expires_in": (
            request.expires_at - int(time.time()) if request.expires_at else None
        ),
    }
    try:
        # The ID token's audience must be the backend's own app, which is also
        # the only client that can renew the refresh token.
        return await credentials_from_tokens(
            payload,
            client_id=microsoft_app().client_id,
            auth_flow="authorization_code",
        )
    except jwt.InvalidAudienceError:
        raise SignInClientMismatch(
            "Microsoft sign-in uses a different app than Microsoft connections."
        ) from None
    except (jwt.InvalidTokenError, KeyError, ValueError) as error:
        raise SignInConnectionError(
            f"Microsoft's sign-in tokens aren't usable: {type(error).__name__}"
        ) from None


async def keep_sign_in_connection(
    user_id: str, credentials: OAuth2Credentials
) -> SignInConnectionResult:
    """Creates the account's connection, or widens the one it has.

    Unlike a Connect, a sign-in never adds a second connection for an account:
    one with more than this sign-in granted (Teams on a work account, say)
    stays as it is. A token without a refresh token (Google issues one only
    on a consent screen) can widen an existing connection, which keeps its
    own refresh token, but can't start one.
    """
    if credentials.username is None and credentials.provider != ProviderName.MICROSOFT:
        # Nothing names the account, so every sign-in would add another one.
        return "skipped"
    existing = [
        saved
        for saved in await creds_manager.store.get_creds_by_provider(
            user_id, credentials.provider
        )
        if isinstance(saved, OAuth2Credentials)
        and not saved.is_managed
        and not is_system_credential(saved.id)
        and _same_account(saved, credentials)
    ]
    for saved in existing:
        if set(credentials.scopes).issuperset(saved.scopes):
            await _upgrade_existing_credential(user_id, saved.id, credentials)
            return "updated"
    if existing:
        return "kept"
    if not credentials.refresh_token:
        return "skipped"
    await creds_manager.create(user_id, credentials)
    return "created"
