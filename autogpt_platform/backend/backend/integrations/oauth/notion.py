import time
from base64 import b64encode
from typing import ClassVar, Optional
from urllib.parse import urlencode

from backend.data.model import OAuth2Credentials
from backend.integrations.providers import ProviderName
from backend.util.request import Requests

from .base import BaseOAuthHandler


class NotionOAuthHandler(BaseOAuthHandler):
    """
    Based on the documentation at https://developers.notion.com/docs/authorization

    Notes:
    - Notion documents no access-token expiry, but returns a refresh token, and each
      refresh replaces both tokens
    - Notion doesn't use scopes
    """

    PROVIDER_NAME = ProviderName.NOTION
    ROTATES_REFRESH_TOKEN: ClassVar[bool] = True

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

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        return await self._request_tokens(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
            }
        )

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        # Notion doesn't support token revocation
        return False

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            return credentials
        return await self._request_tokens(
            {
                "grant_type": "refresh_token",
                "refresh_token": credentials.refresh_token.get_secret_value(),
            },
            current_credentials=credentials,
        )

    async def _request_tokens(
        self,
        request_body: dict[str, str],
        current_credentials: Optional[OAuth2Credentials] = None,
    ) -> OAuth2Credentials:
        auth_str = b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth_str}",
            "Accept": "application/json",
        }
        response = await Requests().post(
            self.token_url, json=request_body, headers=headers
        )
        token_data = response.json()
        # Email is only available for non-bot users
        email = (
            token_data["owner"]["person"]["email"]
            if "person" in token_data["owner"]
            and "email" in token_data["owner"]["person"]
            else None
        )

        expires_in = token_data.get("expires_in")
        new_credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=(
                current_credentials.title
                if current_credentials
                else token_data.get("workspace_name")
            ),
            username=email,
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token")
            or (current_credentials.refresh_token if current_credentials else None),
            access_token_expires_at=(
                int(time.time()) + expires_in if expires_in else None
            ),
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
        if current_credentials:
            new_credentials.id = current_credentials.id
        return new_credentials
