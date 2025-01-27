import urllib.parse
from typing import ClassVar, Optional

import requests

from backend.data.model import OAuth2Credentials, ProviderName
from backend.integrations.oauth.base import BaseOAuthHandler


class TodoistOAuthHandler(BaseOAuthHandler):
    PROVIDER_NAME = ProviderName.TODOIST
    DEFAULT_SCOPES: ClassVar[list[str]] = [
        "task:add",
        "data:read",
        "data:read_write",
        "data:delete",
        "project:delete",
    ]

    AUTHORIZE_URL = "https://todoist.com/oauth/authorize"
    TOKEN_URL = "https://todoist.com/oauth/access_token"

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        params = {
            "client_id": self.client_id,
            "scope": ",".join(self.DEFAULT_SCOPES),
            "state": state,
        }

        return f"{self.AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        """Exchange authorization code for access tokens"""

        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.redirect_uri,
        }

        response = requests.post(self.TOKEN_URL, data=data)
        response.raise_for_status()

        tokens = response.json()

        response = requests.post(
            "https://api.todoist.com/sync/v9/sync",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            data={"sync_token": "*", "resource_types": '["user"]'},
        )
        response.raise_for_status()
        user_info = response.json()
        user_email = user_info["user"].get("email")

        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=None,
            username=user_email,
            access_token=tokens["access_token"],
            refresh_token=None,
            access_token_expires_at=None,
            refresh_token_expires_at=None,
            scopes=scopes,
        )

    def _refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
        # Todoist does not support token refresh
        return credentials

    def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        return False
