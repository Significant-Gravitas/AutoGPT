import time
from typing import Optional
from urllib.parse import urlencode

from backend.data.model import OAuth2Credentials
from backend.integrations.providers import ProviderName
from backend.util.request import Requests

from .base import BaseOAuthHandler


class DiscordOAuthHandler(BaseOAuthHandler):
    """
    Discord OAuth2 handler implementation.

    Based on the documentation at:
    - https://discord.com/developers/docs/topics/oauth2

    Discord OAuth2 tokens expire after 7 days by default and include refresh tokens.
    """

    PROVIDER_NAME = ProviderName.DISCORD
    DEFAULT_SCOPES = ["identify"]  # Basic user information

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_base_url = "https://discord.com/oauth2/authorize"
        self.token_url = "https://discord.com/api/oauth2/token"
        self.revoke_url = "https://discord.com/api/oauth2/token/revoke"

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        # Handle default scopes
        scopes = self.handle_default_scopes(scopes)

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": state,
        }

        # Discord supports PKCE
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        return f"{self.auth_base_url}?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        params = {
            "code": code,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }

        # Include PKCE verifier if provided
        if code_verifier:
            params["code_verifier"] = code_verifier

        return await self._request_tokens(params)

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        if not credentials.access_token:
            raise ValueError("No access token to revoke")

        # Discord requires client authentication for token revocation
        data = {
            "token": credentials.access_token.get_secret_value(),
            "token_type_hint": "access_token",
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        response = await Requests().post(
            url=self.revoke_url,
            data=data,
            headers=headers,
            auth=(self.client_id, self.client_secret),
        )

        # Discord returns 200 OK for successful revocation
        return response.status == 200

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            return credentials

        return await self._request_tokens(
            {
                "refresh_token": credentials.refresh_token.get_secret_value(),
                "grant_type": "refresh_token",
            },
            current_credentials=credentials,
        )

    async def _request_tokens(
        self,
        params: dict[str, str],
        current_credentials: Optional[OAuth2Credentials] = None,
    ) -> OAuth2Credentials:
        request_body = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            **params,
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        response = await Requests().post(
            self.token_url, data=request_body, headers=headers
        )
        token_data: dict = response.json()

        # Get username if this is a new token request
        username = None
        if "access_token" in token_data:
            username = await self._request_username(token_data["access_token"])

        now = int(time.time())
        new_credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=current_credentials.title if current_credentials else None,
            username=username,
            access_token=token_data["access_token"],
            scopes=token_data.get("scope", "").split()
            or (current_credentials.scopes if current_credentials else []),
            refresh_token=token_data.get("refresh_token"),
            # Discord tokens expire after expires_in seconds (typically 7 days)
            access_token_expires_at=(
                now + expires_in
                if (expires_in := token_data.get("expires_in", None))
                else None
            ),
            # Discord doesn't provide separate refresh token expiration
            refresh_token_expires_at=None,
        )

        if current_credentials:
            new_credentials.id = current_credentials.id

        return new_credentials

    async def _request_username(self, access_token: str) -> str | None:
        """
        Fetch the username using the Discord OAuth2 @me endpoint.
        """
        url = "https://discord.com/api/oauth2/@me"
        headers = {
            "Authorization": f"Bearer {access_token}",
        }

        response = await Requests().get(url, headers=headers)

        if not response.ok:
            return None

        # Get user info from the response
        data = response.json()
        user_info = data.get("user", {})

        # Return username (without discriminator)
        return user_info.get("username")
