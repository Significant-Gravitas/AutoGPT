"""
MCP OAuth handler for MCP servers that use OAuth 2.1 authorization.

Unlike other OAuth handlers (GitHub, Google, etc.) where endpoints are fixed,
MCP servers have dynamic endpoints discovered via RFC 9728 / RFC 8414 metadata.
This handler accepts those endpoints at construction time.
"""

import logging
import time
import urllib.parse
from typing import ClassVar, Optional

from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.base import BaseOAuthHandler
from backend.integrations.providers import ProviderName
from backend.util.request import Requests

logger = logging.getLogger(__name__)


class MCPOAuthHandler(BaseOAuthHandler):
    """
    OAuth handler for MCP servers with dynamically-discovered endpoints.

    Construction requires the authorization and token endpoint URLs,
    which are obtained via MCP OAuth metadata discovery
    (``MCPClient.discover_auth`` + ``discover_auth_server_metadata``).
    """

    PROVIDER_NAME: ClassVar[ProviderName | str] = ProviderName.MCP
    DEFAULT_SCOPES: ClassVar[list[str]] = []

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        *,
        authorize_url: str,
        token_url: str,
        revoke_url: str | None = None,
        resource_url: str | None = None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.authorize_url = authorize_url
        self.token_url = token_url
        self.revoke_url = revoke_url
        self.resource_url = resource_url

    def get_login_url(
        self,
        scopes: list[str],
        state: str,
        code_challenge: Optional[str],
    ) -> str:
        scopes = self.handle_default_scopes(scopes)

        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "state": state,
        }
        if scopes:
            params["scope"] = " ".join(scopes)
        # PKCE (S256) — included when the caller provides a code_challenge
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"
        # MCP spec requires resource indicator (RFC 8707)
        if self.resource_url:
            params["resource"] = self.resource_url

        return f"{self.authorize_url}?{urllib.parse.urlencode(params)}"

    async def exchange_code_for_tokens(
        self,
        code: str,
        scopes: list[str],
        code_verifier: Optional[str],
    ) -> OAuth2Credentials:
        data: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret
        if code_verifier:
            data["code_verifier"] = code_verifier
        if self.resource_url:
            data["resource"] = self.resource_url

        response = await Requests(raise_for_status=True).post(
            self.token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        tokens = response.json()

        if "error" in tokens:
            raise RuntimeError(
                f"Token exchange failed: {tokens.get('error_description', tokens['error'])}"
            )

        if "access_token" not in tokens:
            raise RuntimeError("OAuth token response missing 'access_token' field")

        now = int(time.time())
        expires_in = tokens.get("expires_in")

        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=None,
            access_token=SecretStr(tokens["access_token"]),
            refresh_token=(
                SecretStr(tokens["refresh_token"])
                if tokens.get("refresh_token")
                else None
            ),
            access_token_expires_at=now + expires_in if expires_in else None,
            refresh_token_expires_at=None,
            scopes=scopes,
            metadata={
                "mcp_token_url": self.token_url,
                "mcp_resource_url": self.resource_url,
            },
        )

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            raise ValueError("No refresh token available for MCP OAuth credentials")

        data: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token.get_secret_value(),
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret
        if self.resource_url:
            data["resource"] = self.resource_url

        response = await Requests(raise_for_status=True).post(
            self.token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        tokens = response.json()

        if "error" in tokens:
            raise RuntimeError(
                f"Token refresh failed: {tokens.get('error_description', tokens['error'])}"
            )

        if "access_token" not in tokens:
            raise RuntimeError("OAuth refresh response missing 'access_token' field")

        now = int(time.time())
        expires_in = tokens.get("expires_in")

        return OAuth2Credentials(
            id=credentials.id,
            provider=self.PROVIDER_NAME,
            title=credentials.title,
            access_token=SecretStr(tokens["access_token"]),
            refresh_token=(
                SecretStr(tokens["refresh_token"])
                if tokens.get("refresh_token")
                else credentials.refresh_token
            ),
            access_token_expires_at=now + expires_in if expires_in else None,
            refresh_token_expires_at=credentials.refresh_token_expires_at,
            scopes=credentials.scopes,
            metadata=credentials.metadata,
        )

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        if not self.revoke_url:
            return False

        try:
            data = {
                "token": credentials.access_token.get_secret_value(),
                "token_type_hint": "access_token",
                "client_id": self.client_id,
            }
            await Requests().post(
                self.revoke_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return True
        except Exception:
            logger.warning("Failed to revoke MCP OAuth tokens", exc_info=True)
            return False
