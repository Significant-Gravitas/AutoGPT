import time
import urllib.parse
from typing import ClassVar, Optional

import requests

from backend.data.model import OAuth2Credentials, ProviderName
from backend.integrations.oauth.base import BaseOAuthHandler


class TwitterOAuthHandler(BaseOAuthHandler):
    PROVIDER_NAME = ProviderName.TWITTER
    DEFAULT_SCOPES: ClassVar[list[str]] = [
        "tweet.read",
        "tweet.write",
        "tweet.moderate.write",
        "users.read",
        "follows.read",
        "follows.write",
        "offline.access",
        "space.read",
        "mute.read",
        "mute.write",
        "like.read",
        "like.write",
        "list.read",
        "list.write",
        "block.read",
        "block.write",
        "bookmark.read",
        "bookmark.write",
    ]

    AUTHORIZE_URL = "https://twitter.com/i/oauth2/authorize"
    TOKEN_URL = "https://api.x.com/2/oauth2/token"
    USERNAME_URL = "https://api.x.com/2/users/me"
    REVOKE_URL = "https://api.x.com/2/oauth2/revoke"

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        """Generate Twitter OAuth 2.0 authorization URL"""
        # scopes = self.handle_default_scopes(scopes)

        if code_challenge is None:
            raise ValueError("code_challenge is required for Twitter OAuth")

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.DEFAULT_SCOPES),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        return f"{self.AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        """Exchange authorization code for access tokens"""

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        data = {
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri,
            "code_verifier": code_verifier,
        }

        auth = (self.client_id, self.client_secret)

        response = requests.post(self.TOKEN_URL, headers=headers, data=data, auth=auth)
        response.raise_for_status()

        tokens = response.json()

        username = self._get_username(tokens["access_token"])

        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=None,
            username=username,
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token"),
            access_token_expires_at=int(time.time()) + tokens["expires_in"],
            refresh_token_expires_at=None,
            scopes=scopes,
        )

    def _get_username(self, access_token: str) -> str:
        """Get the username from the access token"""
        headers = {"Authorization": f"Bearer {access_token}"}

        params = {"user.fields": "username"}

        response = requests.get(
            f"{self.USERNAME_URL}?{urllib.parse.urlencode(params)}", headers=headers
        )
        response.raise_for_status()

        return response.json()["data"]["username"]

    def _refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
        """Refresh access tokens using refresh token"""
        if not credentials.refresh_token:
            raise ValueError("No refresh token available")

        header = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token.get_secret_value(),
        }

        auth = (self.client_id, self.client_secret)

        response = requests.post(self.TOKEN_URL, headers=header, data=data, auth=auth)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print("HTTP Error:", e)
            print("Response Content:", response.text)
            raise

        tokens = response.json()

        username = self._get_username(tokens["access_token"])

        return OAuth2Credentials(
            id=credentials.id,
            provider=self.PROVIDER_NAME,
            title=None,
            username=username,
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            access_token_expires_at=int(time.time()) + tokens["expires_in"],
            scopes=credentials.scopes,
            refresh_token_expires_at=None,
        )

    def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        """Revoke the access token"""

        header = {"Content-Type": "application/x-www-form-urlencoded"}

        data = {
            "token": credentials.access_token.get_secret_value(),
            "token_type_hint": "access_token",
        }

        auth = (self.client_id, self.client_secret)

        response = requests.post(self.REVOKE_URL, headers=header, data=data, auth=auth)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print("HTTP Error:", e)
            print("Response Content:", response.text)
            raise

        return response.status_code == 200
