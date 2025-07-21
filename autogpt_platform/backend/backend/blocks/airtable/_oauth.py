"""
Airtable OAuth handler implementation.
"""

import base64
import hashlib
import json
import secrets
import time
from typing import Optional
from urllib.parse import urlencode

from backend.sdk import BaseOAuthHandler, OAuth2Credentials, ProviderName, Requests


class AirtableAPIException(Exception):
    """Exception for Airtable API errors."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class AirtableOAuthHandler(BaseOAuthHandler):
    """
    OAuth2 handler for Airtable with PKCE support.
    """

    PROVIDER_NAME = ProviderName("airtable")

    def __init__(self, client_id: str, client_secret: Optional[str], redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_base_url = "https://airtable.com/oauth2/v1/authorize"
        self.token_url = "https://airtable.com/oauth2/v1/token"

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        # Generate code_challenge if not provided (PKCE is required)
        if not code_challenge:
            code_verifier = self._generate_code_verifier()
            code_challenge = self._generate_code_challenge(code_verifier)

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),  # Space-separated for Airtable
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return f"{self.auth_base_url}?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        if not code_verifier:
            raise ValueError("code_verifier is required for Airtable OAuth")

        return await self._request_tokens(
            {
                "code": code,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code",
                "code_verifier": code_verifier,
            }
        )

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        # Airtable doesn't provide a token revocation endpoint
        # Tokens expire naturally after 60 minutes
        return True

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            raise ValueError("No refresh token available")

        return await self._request_tokens(
            {
                "grant_type": "refresh_token",
                "refresh_token": credentials.refresh_token.get_secret_value(),
            },
            current_credentials=credentials,
        )

    async def _request_tokens(
        self,
        params: dict[str, str],
        current_credentials: Optional[OAuth2Credentials] = None,
    ) -> OAuth2Credentials:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        # Add client_id to the request body
        request_body = {
            "client_id": self.client_id,
            **params,
        }

        # Add authorization header if client_secret exists
        if self.client_secret:
            credentials = base64.urlsafe_b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"

        response = await Requests().post(
            self.token_url, data=request_body, headers=headers
        )

        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get(
                    "error_description", error_data.get("error", "Unknown error")
                )
            except json.JSONDecodeError:
                error_message = response.text
            raise AirtableAPIException(
                f"Failed to fetch Airtable tokens ({response.status}): {error_message}",
                response.status,
            )

        token_data = response.json()

        # Calculate expiration times based on expires_in values
        now = int(time.time())
        access_token_expires_at = now + token_data.get("expires_in", 3600)
        refresh_token_expires_at = None
        if token_data.get("refresh_expires_in"):
            refresh_token_expires_at = now + token_data["refresh_expires_in"]

        new_credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=current_credentials.title if current_credentials else None,
            username=await self._request_username(token_data["access_token"])
            or "Unknown User",
            access_token=token_data["access_token"],
            scopes=token_data["scope"].split(" "),  # Space-separated scopes
            refresh_token=token_data.get("refresh_token"),
            access_token_expires_at=access_token_expires_at,
            refresh_token_expires_at=refresh_token_expires_at,
        )

        if current_credentials:
            new_credentials.id = current_credentials.id

        return new_credentials

    async def _request_username(self, access_token: str) -> Optional[str]:
        """Fetch the user's information from Airtable."""
        headers = {
            "Authorization": f"Bearer {access_token}",
        }

        try:
            response = await Requests().get(
                "https://api.airtable.com/v0/meta/whoami",
                headers=headers,
            )

            if response.ok:
                user_data = response.json()
                # Airtable returns user email in the whoami endpoint
                return user_data.get("email", "Unknown User")
            else:
                return None

        except Exception as e:
            print(f"Error fetching username: {e}")
            return None

    def _generate_code_verifier(self) -> str:
        """Generate a code_verifier for PKCE."""
        # 43-128 characters from the set: a-z, A-Z, 0-9, -, ., _, ~
        return (
            base64.urlsafe_b64encode(secrets.token_bytes(96))
            .decode("utf-8")
            .rstrip("=")
        )

    def _generate_code_challenge(self, code_verifier: str) -> str:
        """Generate code_challenge from code_verifier using S256 method."""
        digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
