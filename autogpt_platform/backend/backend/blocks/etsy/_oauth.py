"""Etsy OAuth 2.0 (authorization code with PKCE)."""

import logging
import time
from typing import ClassVar
from urllib.parse import urlencode

from pydantic import BaseModel

from backend.sdk import BaseOAuthHandler, OAuth2Credentials, ProviderName, SecretStr

from ._api import EtsyClient, api_key_header, raise_for_etsy_error

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://www.etsy.com/oauth/connect"
TOKEN_URL = "https://api.etsy.com/v3/public/oauth/token"


class _TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int


class EtsyOAuthHandler(BaseOAuthHandler):
    PROVIDER_NAME = ProviderName("etsy")
    # Each refresh hands back a new refresh token; refresh under the lock.
    ROTATES_REFRESH_TOKEN: ClassVar[bool] = True
    DEFAULT_SCOPES: ClassVar[list[str]] = ["shops_r", "listings_r", "listings_w"]

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: str | None
    ) -> str:
        if not code_challenge:
            raise ValueError("Etsy requires PKCE: no code challenge was provided")
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.handle_default_scopes(scopes)),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return f"{AUTHORIZE_URL}?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: str | None
    ) -> OAuth2Credentials:
        if not code_verifier:
            raise ValueError("Etsy requires PKCE: no code verifier was provided")
        tokens = await self._request_tokens(
            {
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "code": code,
                "code_verifier": code_verifier,
            }
        )
        credentials = self._credentials(tokens, self.handle_default_scopes(scopes))
        credentials.username = await self._shop_name(credentials)
        return credentials

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if credentials.refresh_token is None:
            raise ValueError("This Etsy connection has no refresh token")
        tokens = await self._request_tokens(
            {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": credentials.refresh_token.get_secret_value(),
            }
        )
        refreshed = self._credentials(tokens, credentials.scopes)
        refreshed.id = credentials.id
        refreshed.title = credentials.title
        refreshed.username = credentials.username
        return refreshed

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        # Etsy has no revocation endpoint; access tokens lapse after an hour.
        return False

    async def _request_tokens(self, form: dict[str, str]) -> _TokenResponse:
        # Once: an authorization code is single-use, and a refresh rotates
        # the refresh token, so a retry after a lost reply would fail anyway.
        response = await EtsyClient.raw_requests(attempts=1).post(
            TOKEN_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                **api_key_header(self.client_id, self.client_secret),
            },
            data=form,
        )
        raise_for_etsy_error(response)
        return _TokenResponse.model_validate(response.json())

    def _credentials(
        self, tokens: _TokenResponse, scopes: list[str]
    ) -> OAuth2Credentials:
        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            access_token=SecretStr(tokens.access_token),
            refresh_token=SecretStr(tokens.refresh_token),
            access_token_expires_at=int(time.time()) + tokens.expires_in,
            # Etsy keeps a refresh token working for 90 days.
            refresh_token_expires_at=int(time.time()) + 90 * 24 * 3600,
            scopes=scopes,
        )

    async def _shop_name(self, credentials: OAuth2Credentials) -> str | None:
        """The shop's name, so the account list says which shop this is."""
        client = EtsyClient(credentials, self.client_id, self.client_secret)
        try:
            return (await client.get_my_shop()).shop_name
        except Exception as e:
            # Only a label: the code is spent by now, so no failure here
            # (HTTP, transport or an odd reply) may cost the user the sign-in.
            logger.warning(f"Etsy shop name lookup failed: {e}")
            return None
