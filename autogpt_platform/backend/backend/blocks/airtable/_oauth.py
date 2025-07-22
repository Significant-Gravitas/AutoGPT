"""
Airtable OAuth handler implementation.
"""

import base64
import hashlib
import secrets
import time
from typing import Optional

from backend.sdk import BaseOAuthHandler, OAuth2Credentials, ProviderName, SecretStr

from ._api import (
    OAuthTokenResponse,
    oauth_authorize,
    oauth_exchange_code_for_tokens,
    oauth_refresh_tokens,
)
from ._config import AirtableScope


class AirtableOAuthHandler(BaseOAuthHandler):
    """
    OAuth2 handler for Airtable with PKCE support.
    """

    PROVIDER_NAME = ProviderName("airtable")
    DEFAULT_SCOPES = [
        AirtableScope.DATA_RECORDS_READ,
        AirtableScope.DATA_RECORDS_WRITE,
        AirtableScope.SCHEMA_BASES_READ,
        AirtableScope.SCHEMA_BASES_WRITE,
        AirtableScope.WEBHOOK_MANAGE,
    ]

    def __init__(self, client_id: str, client_secret: Optional[str], redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.code_verifier = self._generate_code_verifier()
        self.scopes = self.DEFAULT_SCOPES
        self.auth_base_url = "https://airtable.com/oauth2/v1/authorize"
        self.token_url = "https://airtable.com/oauth2/v1/token"

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        # Generate code_challenge if not provided (PKCE is required)
        code_challenge = self._generate_code_challenge()
        if scopes:
            self.scopes = scopes

        return oauth_authorize(
            self.client_id, self.redirect_uri, self.scopes, state, code_challenge
        )

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        response: OAuthTokenResponse = await oauth_exchange_code_for_tokens(
            client_id=self.client_id,
            code=code,
            code_verifier=self.code_verifier,
            redirect_uri=self.redirect_uri,
            client_secret=self.client_secret,
        )

        return OAuth2Credentials(
            access_token=SecretStr(response.access_token),
            refresh_token=SecretStr(response.refresh_token),
            access_token_expires_at=int(time.time()) + response.expires_in,
            refresh_token_expires_at=int(time.time()) + response.refresh_expires_in,
            provider=self.PROVIDER_NAME,
            scopes=scopes,
        )

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if credentials.refresh_token is None:
            raise ValueError("No refresh token available")

        response: OAuthTokenResponse = await oauth_refresh_tokens(
            client_id=self.client_id,
            refresh_token=credentials.refresh_token.get_secret_value(),
            client_secret=self.client_secret,
        )

        return OAuth2Credentials(
            access_token=SecretStr(response.access_token),
            refresh_token=SecretStr(response.refresh_token),
            access_token_expires_at=int(time.time()) + response.expires_in,
            refresh_token_expires_at=int(time.time()) + response.refresh_expires_in,
            provider=self.PROVIDER_NAME,
            scopes=self.scopes,
        )

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        # Airtable doesn't provide a token revocation endpoint
        # Tokens expire naturally after 60 minutes
        return False

    def _generate_code_verifier(self) -> bytes:
        """Generate a code_verifier for PKCE."""
        # 43-128 characters from the set: a-z, A-Z, 0-9, -, ., _, ~
        return base64.urlsafe_b64encode(secrets.token_bytes(96)).rstrip(b"=")

    def _generate_code_challenge(self) -> str:
        """Generate code_challenge from code_verifier using S256 method."""
        digest = hashlib.sha256(self.code_verifier).digest()
        return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
