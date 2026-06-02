import time
import urllib.parse
from typing import ClassVar, Optional

from backend.data.model import OAuth2Credentials, ProviderName
from backend.integrations.oauth.base import BaseOAuthHandler
from backend.util.request import Requests


class LinkedInOAuthHandler(BaseOAuthHandler):
    PROVIDER_NAME = ProviderName.LINKEDIN
    DEFAULT_SCOPES: ClassVar[list[str]] = [
        "openid",
        "profile",
        "email",
        "w_member_social",
        "r_basicprofile",
        "r_1st_connections_size",
    ]

    AUTHORIZE_URL = "https://www.linkedin.com/oauth/v2/authorization"
    TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
    USERINFO_URL = "https://api.linkedin.com/v2/userinfo"

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.DEFAULT_SCOPES),
            "state": state,
        }
        return f"{self.AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = await Requests().post(self.TOKEN_URL, headers=headers, data=data)
        tokens = response.json()
        username = await self._get_username(tokens["access_token"])
        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=None,
            username=username,
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token"),
            access_token_expires_at=int(time.time()) + tokens.get("expires_in", 3600),
            refresh_token_expires_at=None,
            scopes=scopes,
        )

    async def _get_username(self, access_token: str) -> str:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await Requests().get(self.USERINFO_URL, headers=headers)
        data = response.json()
        return data.get("email") or data.get("sub", "linkedin-user")

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            raise ValueError("No refresh token available")
        data = {
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token.get_secret_value(),
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = await Requests().post(self.TOKEN_URL, headers=headers, data=data)
        if not response.ok:
            raise ValueError(f"Token refresh failed: {response.status_code}")
        tokens = response.json()
        username = await self._get_username(tokens["access_token"])
        return OAuth2Credentials(
            id=credentials.id,
            provider=self.PROVIDER_NAME,
            title=None,
            username=username,
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token", credentials.refresh_token),
            access_token_expires_at=int(time.time()) + tokens.get("expires_in", 3600),
            refresh_token_expires_at=None,
            scopes=credentials.scopes,
        )

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        # LinkedIn does not provide a token revocation endpoint
        return True
