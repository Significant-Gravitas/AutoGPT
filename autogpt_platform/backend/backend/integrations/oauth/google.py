import asyncio
import logging
import time
from typing import Optional

import aiohttp
from google.auth.exceptions import RefreshError
from google.auth.external_account_authorized_user import (
    Credentials as ExternalAccountCredentials,
)
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.providers import ProviderName, provider_key
from backend.util.request import Requests

from .base import BaseOAuthHandler
from .refresh_failure import OAuthTokenRequestError, describe_refresh_failure
from .sign_in import SignInClientMismatch, SignInConnectionError

logger = logging.getLogger(__name__)

# Under the frontend's 8 s wait for a sign-in's handoff, retries included, so
# this check never outlives the request that asked for it.
TOKEN_INFO_TIMEOUT_SECONDS = 5


def _expires_at(expires_in: object) -> int | None:
    try:
        return int(time.time()) + int(str(expires_in))
    except ValueError:
        return None


class _StatusRecordingRequest(Request):
    """google-auth's transport, remembering the last HTTP status it saw.

    ``RefreshError`` carries Google's error body but not the status, and the
    status is half of what a failed refresh has to report.
    """

    last_status: int | None = None

    def __call__(self, *args, **kwargs):
        response = super().__call__(*args, **kwargs)
        self.last_status = response.status
        return response


# --8<-- [start:GoogleOAuthHandlerExample]
class GoogleOAuthHandler(BaseOAuthHandler):
    """
    Based on the documentation at https://developers.google.com/identity/protocols/oauth2/web-server
    """  # noqa

    PROVIDER_NAME = ProviderName.GOOGLE
    EMAIL_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"
    DEFAULT_SCOPES = [
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "openid",
    ]
    # --8<-- [end:GoogleOAuthHandlerExample]
    TOKEN_INFO_ENDPOINT = "https://oauth2.googleapis.com/tokeninfo"

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.token_uri = "https://oauth2.googleapis.com/token"
        self.revoke_uri = "https://oauth2.googleapis.com/revoke"

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        all_scopes = list(set(scopes + self.DEFAULT_SCOPES))
        logger.debug(f"Setting up OAuth flow with scopes: {all_scopes}")
        flow = self._setup_oauth_flow(all_scopes)
        flow.redirect_uri = self.redirect_uri
        # The platform generated this challenge together with the verifier it
        # stores on the OAuth state, so the challenge on the login URL is the
        # one the verifier in exchange_code_for_tokens can answer for.
        pkce_params = (
            {"code_challenge": code_challenge, "code_challenge_method": "S256"}
            if code_challenge
            else {}
        )
        authorization_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            state=state,
            prompt="consent",
            **pkce_params,
        )
        return authorization_url

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        logger.debug(f"Exchanging code for tokens with scopes: {scopes}")

        # Use the scopes from the initial request
        flow = self._setup_oauth_flow(scopes)
        flow.redirect_uri = self.redirect_uri

        logger.debug("Fetching token from Google")

        # Disable scope check in fetch_token
        flow.oauth2session.scope = None
        # This is a different Flow than the one that built the login URL, so
        # the verifier has to come from the platform's OAuth state.
        token = flow.fetch_token(code=code, code_verifier=code_verifier)
        logger.debug("Token fetched successfully")

        # Get the actual scopes granted by Google
        granted_scopes: list[str] = token.get("scope", [])

        logger.debug(f"Scopes granted by Google: {granted_scopes}")

        google_creds = flow.credentials
        logger.debug("Received credentials")

        logger.debug("Requesting user email")
        username = self._request_email(google_creds)
        logger.debug(f"User email retrieved: {username}")

        assert google_creds.token
        assert google_creds.refresh_token
        assert google_creds.expiry
        assert granted_scopes

        # Create OAuth2Credentials with the granted scopes
        credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=None,
            username=username,
            access_token=SecretStr(google_creds.token),
            refresh_token=(SecretStr(google_creds.refresh_token)),
            access_token_expires_at=(
                int(google_creds.expiry.timestamp()) if google_creds.expiry else None
            ),
            refresh_token_expires_at=None,
            scopes=granted_scopes,
        )
        logger.debug(
            f"OAuth2Credentials object created successfully with scopes: {credentials.scopes}"
        )

        return credentials

    async def credentials_from_sign_in(
        self,
        access_token: str,
        refresh_token: str | None,
        expires_at: int | None,
    ) -> OAuth2Credentials:
        """The connection a Google sign-in's tokens make.

        Google's tokeninfo says which OAuth client issued the token, what it
        grants and whose it is. Only this handler's own client is accepted:
        Google renews a refresh token only for the client it was issued to.
        """
        # In the body, not the query string, so the token stays out of URLs.
        # One deadline for the whole check: a throttled answer is retried.
        try:
            response = await asyncio.wait_for(
                Requests(raise_for_status=False, retry_max_attempts=3).post(
                    self.TOKEN_INFO_ENDPOINT,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={"access_token": access_token},
                ),
                timeout=TOKEN_INFO_TIMEOUT_SECONDS,
            )
        # ValueError: Requests' own refusal, such as a hostname DNS can't
        # resolve, before anything is sent.
        except (asyncio.TimeoutError, aiohttp.ClientError, ValueError):
            raise SignInConnectionError(
                "Google didn't answer the sign-in's token check."
            ) from None
        if not response.ok:
            raise SignInConnectionError("Google didn't accept the sign-in's token.")
        try:
            info = response.json()
        except ValueError:
            info = None
        if not isinstance(info, dict):
            raise SignInConnectionError(
                "Google's token check answered with no token info."
            )
        if self.client_id not in (info.get("azp"), info.get("aud")):
            raise SignInClientMismatch(
                "Google sign-in uses a different OAuth client than Google "
                "connections."
            )
        email = info.get("email")
        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=None,
            username=email if isinstance(email, str) and email else None,
            access_token=SecretStr(access_token),
            refresh_token=SecretStr(refresh_token) if refresh_token else None,
            access_token_expires_at=expires_at or _expires_at(info.get("expires_in")),
            refresh_token_expires_at=None,
            scopes=str(info.get("scope", "")).split(),
        )

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        # Google's revoke endpoint only needs the token in the request body;
        # it does not need an Authorization header. Previously this used
        # google-auth's AuthorizedSession, which calls .before_request() on
        # the credential object — that fails with AttributeError when given
        # our Pydantic OAuth2Credentials (which isn't a google-auth
        # Credentials). Switching to the platform's async Requests helper
        # matches how the other providers (reddit/github/...) do it.
        # A refresh token revokes the whole grant. A stored access token has
        # usually expired (they live 1h and refresh only on use), and Google
        # answers 400 invalid_token for it without revoking anything.
        token = credentials.refresh_token or credentials.access_token
        if not token:
            return False

        # raise_for_status=False so a 400 (already-revoked token, malformed
        # body, rate limit, etc.) flows back as response.ok=False instead of
        # raising — the caller uses the bool to set the UI `revoked` flag
        # and does not need to distinguish failure modes.
        # retry_max_attempts bounds the retries on 429/5xx, so a transient
        # Google failure can't hold up the credential deletion API call.
        response = await Requests(
            raise_for_status=False,
            retry_max_attempts=3,
        ).post(
            self.revoke_uri,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"token": token.get_secret_value()},
        )
        return response.ok

    def _request_email(
        self, creds: Credentials | ExternalAccountCredentials
    ) -> str | None:
        session = AuthorizedSession(creds)
        response = session.get(self.EMAIL_ENDPOINT)
        if not response.ok:
            logger.error(
                f"Failed to get user email. Status code: {response.status_code}"
            )
            return None
        return response.json()["email"]

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        # Google credentials should ALWAYS have a refresh token
        assert credentials.refresh_token

        google_creds = Credentials(
            token=credentials.access_token.get_secret_value(),
            refresh_token=credentials.refresh_token.get_secret_value(),
            token_uri=self.token_uri,
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=credentials.scopes,
        )
        # Google's OAuth library is poorly typed so we need some of these:
        assert google_creds.refresh_token
        assert google_creds.scopes

        request = _StatusRecordingRequest()
        try:
            await asyncio.to_thread(google_creds.refresh, request)
        except RefreshError as e:
            raise OAuthTokenRequestError(
                provider_key(self.PROVIDER_NAME),
                status_code=request.last_status,
                error_code=describe_refresh_failure(e).error_code,
            ) from e
        assert google_creds.expiry

        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            id=credentials.id,
            title=credentials.title,
            username=credentials.username,
            access_token=SecretStr(google_creds.token),
            refresh_token=SecretStr(google_creds.refresh_token),
            access_token_expires_at=int(google_creds.expiry.timestamp()),
            refresh_token_expires_at=None,
            scopes=google_creds.scopes,
        )

    def _setup_oauth_flow(self, scopes: list[str]) -> Flow:
        # google-auth-oauthlib >= 1.3.0 generates its own PKCE verifier per
        # Flow by default. A verifier minted here would only live on this
        # Flow, which is discarded after the login URL is built, so the token
        # exchange could never present it and Google would reject every
        # exchange with "(invalid_grant) Missing code verifier". PKCE is
        # handled with the platform's own pair instead (see get_login_url and
        # exchange_code_for_tokens).
        return Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": self.token_uri,
                }
            },
            scopes=scopes,
            autogenerate_code_verifier=False,
        )
