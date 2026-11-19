"""Connections that come with a Google or Microsoft sign-in.

Login can ask Google or Microsoft for the data scopes too (the frontend's
``lib/auth/connect-at-sign-in.ts``). The frontend's server then hands this
endpoint the tokens from that sign-in, so the person starts with the
connection the Connect flow would have made, without a second consent screen.

Only the frontend's server can call it: its service token names the user who
just signed in. A sign-in never adds a second connection for an account that
already has one: it widens the existing one when the new token covers it, and
otherwise leaves it alone.

Only a sign-in through the OAuth client the connections use can become one,
since only that client can renew its tokens. ``GET`` says which clients those
are, and the frontend asks for the data scopes only when its sign-in uses
the same one. So a deployment whose sign-in has a client of its own, or that
has no Google or Microsoft connections, signs people in as before.

A sign-up's handoff arrives before the account's first request has set it up,
and a connection can only be stored once the account has its personal org. The
connection is held until then instead (encrypted, for ``HELD_FOR_SECONDS``),
and ``POST /auth/user``, the request that sets the account up, keeps it. That
request stays the one that creates the account, so everything it does for a
new account (the signup goal, the consent record) still happens.
"""

import asyncio
import logging
import time
from typing import Annotated, Literal

import jwt
from autogpt_libs.auth.service import frontend_service_claims
from fastapi import APIRouter, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, SecretStr

from backend.data.model import OAuth2Credentials
from backend.data.org_migration import has_personal_org
from backend.data.redis_client import get_redis_async
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
from backend.util.encryption import JSONCryptor
from backend.util.settings import Settings

from .router import (
    _get_provider_oauth_handler,
    _same_account,
    _upgrade_existing_credential,
    creds_manager,
)

logger = logging.getLogger(__name__)

SIGN_IN_CONNECTION_SCOPE = "integrations:sign-in-connection"
# Well past any Google or Microsoft token, so only a malformed body hits it.
TOKEN_MAX_LENGTH = 16_384
# The sign-up's own redirect sets the account up within seconds; a connection
# nobody takes by then is dropped, and onboarding offers to connect instead.
HELD_FOR_SECONDS = 600
# Every sign-in's POST /auth/user looks for a held connection, so an
# unreachable Redis must cost it at most this, never a hung read.
HELD_LOOKUP_TIMEOUT_SECONDS = 2.0

sign_in_connection_router = APIRouter()
_bearer = HTTPBearer(auto_error=False)


class SignInConnectionRequest(BaseModel):
    provider: Literal["google", "microsoft"]
    access_token: SecretStr = Field(max_length=TOKEN_MAX_LENGTH)
    refresh_token: SecretStr | None = Field(default=None, max_length=TOKEN_MAX_LENGTH)
    # Unix seconds.
    expires_at: int | None = None
    # What the provider granted, space-separated, as its token response says.
    scope: str = Field(default="", max_length=4_096)
    # Microsoft names the account in it; Google's isn't needed.
    id_token: SecretStr | None = Field(default=None, max_length=TOKEN_MAX_LENGTH)


SignInConnectionResult = Literal["created", "updated", "kept", "skipped", "held"]


class SignInConnectionResponse(BaseModel):
    result: SignInConnectionResult


class SignInConnectionClients(BaseModel):
    """The OAuth client each provider's connections use, or null when this
    deployment has no such connections."""

    google: str | None = None
    microsoft: str | None = None


@sign_in_connection_router.get(
    "/sign-in-connection",
    summary="The OAuth clients a sign-in must use to bring its connection",
    operation_id="getSignInConnectionClients",
)
async def get_sign_in_connection_clients() -> SignInConnectionClients:
    """Public, like the client IDs themselves: every sign-in link carries one."""
    return SignInConnectionClients(
        google=_google_connection_client(),
        microsoft=_microsoft_connection_client(),
    )


def _google_connection_client() -> str | None:
    # What _get_provider_oauth_handler builds Google's handler from.
    secrets = Settings().secrets
    if secrets.google_client_id and secrets.google_client_secret:
        return secrets.google_client_id
    return None


def _microsoft_connection_client() -> str | None:
    app = microsoft_app()
    # A sign-in's tokens belong to a web app, which renews them only with its
    # secret.
    return app.client_id if app.client_secret else None


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
    if await has_personal_org(user_id):
        result = await keep_sign_in_connection(user_id, credentials)
    else:
        result = await _hold_for_set_up(user_id, credentials)
    logger.info("Sign-in connection for %s: %s", request.provider, result)
    return SignInConnectionResponse(result=result)


async def _hold_for_set_up(
    user_id: str, credentials: OAuth2Credentials
) -> SignInConnectionResult:
    """Holds a sign-up's connection for the request that sets its account up."""
    try:
        await hold_sign_in_connection(user_id, credentials)
    except Exception:
        # Never the error. The sign-in goes on, and onboarding offers the
        # connection instead.
        logger.warning("A sign-up's connection couldn't be held")
        return "skipped"
    # Set up after the check above but before the hold, the account's own
    # look for a held connection came too early: keep it here instead.
    if await has_personal_org(user_id):
        return await keep_held_sign_in_connection(user_id) or "held"
    return "held"


def _held_key(user_id: str) -> str:
    return f"sign-in-connection:{user_id}"


async def hold_sign_in_connection(user_id: str, credentials: OAuth2Credentials) -> None:
    """Keeps a sign-up's connection until its account is set up."""
    sealed = JSONCryptor().encrypt(credentials.model_dump(mode="json"))
    await asyncio.wait_for(
        _put_held(user_id, sealed), timeout=HELD_LOOKUP_TIMEOUT_SECONDS
    )


async def _put_held(user_id: str, sealed: str) -> None:
    redis = await get_redis_async()
    await redis.set(_held_key(user_id), sealed, ex=HELD_FOR_SECONDS)


async def keep_held_sign_in_connection(
    user_id: str,
) -> SignInConnectionResult | None:
    """Keeps the connection held for *user_id*, now that the account is set up.

    Called by ``POST /auth/user`` once the account exists. Takes the held
    connection at most once, and never fails the request it runs in: without
    the connection, onboarding offers to connect. None when nothing was held
    (or Redis couldn't say); ``skipped`` when one was taken but couldn't be
    read or kept, so it's gone.
    """
    try:
        sealed = await asyncio.wait_for(
            _take_held(user_id), timeout=HELD_LOOKUP_TIMEOUT_SECONDS
        )
    except Exception:
        logger.warning("Couldn't read the held sign-in connection")
        return None
    if not sealed:
        return None
    try:
        credentials = OAuth2Credentials.model_validate(JSONCryptor().decrypt(sealed))
    except Exception:
        # Never the error: a validation error quotes its input, the tokens.
        logger.warning("A held sign-in connection couldn't be read")
        return "skipped"
    try:
        result = await keep_sign_in_connection(user_id, credentials)
    except Exception:
        logger.warning("A held sign-in connection wasn't kept")
        return "skipped"
    logger.info("Held sign-in connection for %s: %s", credentials.provider, result)
    return result


async def _take_held(user_id: str) -> str | None:
    redis = await get_redis_async()
    sealed = await redis.getdel(_held_key(user_id))
    # The client decodes responses, which its stubs can't know.
    return sealed.decode() if isinstance(sealed, bytes) else sealed


async def _credentials_from_sign_in(
    request: SignInConnectionRequest,
) -> OAuth2Credentials:
    access_token = request.access_token.get_secret_value()
    refresh_token = (
        request.refresh_token.get_secret_value() if request.refresh_token else None
    )
    if request.provider == "google":
        handler = _get_provider_oauth_handler(ProviderName.GOOGLE)
        if not isinstance(handler, GoogleOAuthHandler):
            raise HTTPException(
                status.HTTP_501_NOT_IMPLEMENTED,
                detail="Integration with provider 'google' is not configured.",
            )
        return await handler.credentials_from_sign_in(
            access_token, refresh_token, request.expires_at
        )
    return await _microsoft_credentials(request, access_token, refresh_token)


async def _microsoft_credentials(
    request: SignInConnectionRequest, access_token: str, refresh_token: str | None
) -> OAuth2Credentials:
    # As for Google's handler: an app without its secret can't renew the
    # tokens, so there is no connection to keep. The frontend asks first, but
    # its answer can be a few minutes old.
    if _microsoft_connection_client() is None:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            detail="Integration with provider 'microsoft' is not configured.",
        )
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
    # PyJWTError, not just InvalidTokenError: Microsoft's signing keys out of
    # reach (PyJWKClientError) means the token can't be checked either.
    except (jwt.PyJWTError, KeyError, ValueError) as error:
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
