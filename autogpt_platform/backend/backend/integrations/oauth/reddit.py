import time
import urllib.parse
from typing import ClassVar, Optional

from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.base import BaseOAuthHandler
from backend.integrations.providers import ProviderName
from backend.util.request import Requests
from backend.util.settings import Settings

settings = Settings()


class RedditOAuthHandler(BaseOAuthHandler):
    """
    Reddit OAuth 2.0 handler.

    Based on the documentation at:
    - https://github.com/reddit-archive/reddit/wiki/OAuth2

    Notes:
    - Reddit requires `duration=permanent` to get refresh tokens
    - Access tokens expire after 1 hour (3600 seconds)
    - Reddit requires HTTP Basic Auth for token requests
    - Reddit requires a unique User-Agent header
    """

    PROVIDER_NAME = ProviderName.REDDIT
    DEFAULT_SCOPES: ClassVar[list[str]] = [
        "identity",  # Get username, verify auth
        "read",  # Access posts and comments
        "submit",  # Submit new posts and comments
        "edit",  # Edit own posts and comments
        "history",  # Access user's post history
        "privatemessages",  # Access inbox and send private messages
        "flair",  # Access and set flair on posts/subreddits
    ]

    AUTHORIZE_URL = "https://www.reddit.com/api/v1/authorize"
    TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
    USERNAME_URL = "https://oauth.reddit.com/api/v1/me"
    REVOKE_URL = "https://www.reddit.com/api/v1/revoke_token"

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        """Generate Reddit OAuth 2.0 authorization URL"""
        scopes = self.handle_default_scopes(scopes)

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(scopes),
            "state": state,
            "duration": "permanent",  # Required for refresh tokens
        }

        return f"{self.AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        """Exchange authorization code for access tokens"""
        scopes = self.handle_default_scopes(scopes)

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": settings.config.reddit_user_agent,
        }

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
        }

        # Reddit requires HTTP Basic Auth for token requests
        auth = (self.client_id, self.client_secret)

        response = await Requests().post(
            self.TOKEN_URL, headers=headers, data=data, auth=auth
        )

        if not response.ok:
            error_text = response.text()
            raise ValueError(
                f"Reddit token exchange failed: {response.status} - {error_text}"
            )

        tokens = response.json()

        if "error" in tokens:
            raise ValueError(f"Reddit OAuth error: {tokens.get('error')}")

        username = await self._get_username(tokens["access_token"])

        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=None,
            username=username,
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token"),
            access_token_expires_at=int(time.time()) + tokens.get("expires_in", 3600),
            refresh_token_expires_at=None,  # Reddit refresh tokens don't expire
            scopes=scopes,
        )

    async def _get_username(self, access_token: str) -> str:
        """Get the username from the access token"""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": settings.config.reddit_user_agent,
        }

        response = await Requests().get(self.USERNAME_URL, headers=headers)

        if not response.ok:
            raise ValueError(f"Failed to get Reddit username: {response.status}")

        data = response.json()
        return data.get("name", "unknown")

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        """Refresh access tokens using refresh token"""
        if not credentials.refresh_token:
            raise ValueError("No refresh token available")

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": settings.config.reddit_user_agent,
        }

        data = {
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token.get_secret_value(),
        }

        auth = (self.client_id, self.client_secret)

        response = await Requests().post(
            self.TOKEN_URL, headers=headers, data=data, auth=auth
        )

        if not response.ok:
            error_text = response.text()
            raise ValueError(
                f"Reddit token refresh failed: {response.status} - {error_text}"
            )

        tokens = response.json()

        if "error" in tokens:
            raise ValueError(f"Reddit OAuth error: {tokens.get('error')}")

        username = await self._get_username(tokens["access_token"])

        # Reddit may or may not return a new refresh token
        new_refresh_token = tokens.get("refresh_token")
        if new_refresh_token:
            refresh_token: SecretStr | None = SecretStr(new_refresh_token)
        elif credentials.refresh_token:
            # Keep the existing refresh token
            refresh_token = credentials.refresh_token
        else:
            refresh_token = None

        return OAuth2Credentials(
            id=credentials.id,
            provider=self.PROVIDER_NAME,
            title=credentials.title,
            username=username,
            access_token=tokens["access_token"],
            refresh_token=refresh_token,
            access_token_expires_at=int(time.time()) + tokens.get("expires_in", 3600),
            refresh_token_expires_at=None,
            scopes=credentials.scopes,
        )

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        """Revoke the access token"""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": settings.config.reddit_user_agent,
        }

        data = {
            "token": credentials.access_token.get_secret_value(),
            "token_type_hint": "access_token",
        }

        auth = (self.client_id, self.client_secret)

        response = await Requests().post(
            self.REVOKE_URL, headers=headers, data=data, auth=auth
        )

        # Reddit returns 204 No Content on successful revocation
        return response.ok
