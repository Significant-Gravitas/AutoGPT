from base64 import b64encode
from typing import Optional
from urllib.parse import urlencode

from backend.data.model import OAuth2Credentials
from backend.integrations.providers import ProviderName
from backend.util.request import requests

from .base import BaseOAuthHandler


class NotionOAuthHandler(BaseOAuthHandler):
    """
    Based on the documentation at https://developers.notion.com/docs/authorization

    Notes:
    - Notion uses non-expiring access tokens and therefore doesn't have a refresh flow
    - Notion doesn't use scopes
    """

    PROVIDER_NAME = ProviderName.NOTION

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_base_url = "https://api.notion.com/v1/oauth/authorize"
        self.token_url = "https://api.notion.com/v1/oauth/token"

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "owner": "user",
            "state": state,
        }
        return f"{self.auth_base_url}?{urlencode(params)}"

    def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        request_body = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
        }
        auth_str = b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth_str}",
            "Accept": "application/json",
        }
        response = requests.post(self.token_url, json=request_body, headers=headers)
        token_data = response.json()
        # Email is only available for non-bot users
        email = (
            token_data["owner"]["person"]["email"]
            if "person" in token_data["owner"]
            and "email" in token_data["owner"]["person"]
            else None
        )

        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=token_data.get("workspace_name"),
            username=email,
            access_token=token_data["access_token"],
            refresh_token=None,
            access_token_expires_at=None,  # Notion tokens don't expire
            refresh_token_expires_at=None,
            scopes=[],
            metadata={
                "owner": token_data["owner"],
                "bot_id": token_data["bot_id"],
                "workspace_id": token_data["workspace_id"],
                "workspace_name": token_data.get("workspace_name"),
                "workspace_icon": token_data.get("workspace_icon"),
            },
        )

    def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        # Notion doesn't support token revocation
        return False

    def _refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
        # Notion doesn't support token refresh
        return credentials

    def needs_refresh(self, credentials: OAuth2Credentials) -> bool:
        # Notion access tokens don't expire
        return False
