import urllib.parse
from typing import ClassVar, Optional

import requests
from backend.data.model import OAuth2Credentials, ProviderName

from backend.integrations.oauth.base import BaseOAuthHandler


class SlackBotOAuthHandler(BaseOAuthHandler):
    PROVIDER_NAME = ProviderName.SLACK_BOT
    DEFAULT_SCOPES: ClassVar[list[str]] = [
        "chat:write",
        "users:read",
    ]

    AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
    TOKEN_URL = "https://slack.com/api/oauth.v2.access"
    USERNAME_URL = "https://slack.com/api/auth.test"
    REVOKE_URL = "https://slack.com/api/auth.revoke"

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        """Generate Slack OAuth 2.0 authorization URL"""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": ",".join(self.DEFAULT_SCOPES),
            "user_scope": ",".join(self.DEFAULT_SCOPES),
            "state": state,
        }

        return f"{self.AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        """Exchange authorization code for access tokens"""

        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
        }

        response = requests.post(self.TOKEN_URL, headers=headers, data=data)
        response.raise_for_status()

        tokens = response.json()

        if not tokens["ok"]:
            raise Exception(f"Failed to get tokens: {tokens['error']}")

        username = self._get_username(tokens["access_token"])

        bot_token = tokens["access_token"]

        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=None,
            username=username,
            access_token= bot_token,
            refresh_token=None,  # Slack doesn't use refresh tokens
            access_token_expires_at=None,  # Slack tokens don't expire
            refresh_token_expires_at=None,
            scopes=scopes
        )

    def _get_username(self, access_token: str) -> str:
        """Get the username from the access token"""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.USERNAME_URL, headers=headers)
        response.raise_for_status()

        data = response.json()
        if not data["ok"]:
            raise Exception(f"Failed to get username: {data['error']}")

        return data["user_id"]

    def _refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
        """Refresh tokens not supported by Slack"""
        return credentials

    def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        """Revoke the access token"""
        headers = {
            "Authorization": f"Bearer {credentials.access_token.get_secret_value()}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.REVOKE_URL, headers=headers)

        try:
            response.raise_for_status()
            data = response.json()
            return data["ok"]
        except requests.exceptions.HTTPError as e:
            print("HTTP Error:", e)
            print("Response Content:", response.text)
            raise

        return False
