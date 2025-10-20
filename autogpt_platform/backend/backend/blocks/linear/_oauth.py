"""
Linear OAuth handler implementation.
"""

import base64
import json
import time
from typing import Optional
from urllib.parse import urlencode

from backend.sdk import (
    APIKeyCredentials,
    BaseOAuthHandler,
    OAuth2Credentials,
    ProviderName,
    Requests,
    SecretStr,
)


class LinearAPIException(Exception):
    """Exception for Linear API errors."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class LinearOAuthHandler(BaseOAuthHandler):
    """
    OAuth2 handler for Linear.
    """

    # Provider name will be set dynamically by the SDK when registered
    # We use a placeholder that will be replaced by AutoRegistry.register_provider()
    PROVIDER_NAME = ProviderName("linear")

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_base_url = "https://linear.app/oauth/authorize"
        self.token_url = "https://api.linear.app/oauth/token"
        self.revoke_url = "https://api.linear.app/oauth/revoke"
        self.migrate_url = "https://api.linear.app/oauth/migrate_old_token"

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",  # Important: include "response_type"
            "scope": ",".join(scopes),  # Comma-separated, not space-separated
            "state": state,
        }
        return f"{self.auth_base_url}?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        return await self._request_tokens(
            {"code": code, "redirect_uri": self.redirect_uri}
        )

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        if not credentials.access_token:
            raise ValueError("No access token to revoke")

        headers = {
            "Authorization": f"Bearer {credentials.access_token.get_secret_value()}"
        }

        response = await Requests().post(self.revoke_url, headers=headers)
        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get("error", "Unknown error")
            except json.JSONDecodeError:
                error_message = response.text
            raise LinearAPIException(
                f"Failed to revoke Linear tokens ({response.status}): {error_message}",
                response.status,
            )

        return True  # Linear doesn't return JSON on successful revoke

    async def migrate_old_token(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        """
        Migrate an old long-lived token to a new short-lived token with refresh token.

        This uses Linear's /oauth/migrate_old_token endpoint to exchange current
        long-lived tokens for short-lived tokens with refresh tokens without
        requiring users to re-authorize.
        """
        if not credentials.access_token:
            raise ValueError("No access token to migrate")

        request_body = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        headers = {
            "Authorization": f"Bearer {credentials.access_token.get_secret_value()}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        response = await Requests().post(
            self.migrate_url, data=request_body, headers=headers
        )

        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get("error", "Unknown error")
                error_description = error_data.get("error_description", "")
                if error_description:
                    error_message = f"{error_message}: {error_description}"
            except json.JSONDecodeError:
                error_message = response.text
            raise LinearAPIException(
                f"Failed to migrate Linear token ({response.status}): {error_message}",
                response.status,
            )

        token_data = response.json()

        # Extract token expiration
        now = int(time.time())
        expires_in = token_data.get("expires_in")
        access_token_expires_at = None
        if expires_in:
            access_token_expires_at = now + expires_in

        new_credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=credentials.title,
            username=credentials.username,
            access_token=token_data["access_token"],
            scopes=credentials.scopes,  # Preserve original scopes
            refresh_token=token_data.get("refresh_token"),
            access_token_expires_at=access_token_expires_at,
            refresh_token_expires_at=None,
        )

        new_credentials.id = credentials.id
        return new_credentials

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            raise ValueError(
                "No refresh token available. Token may need to be migrated to the new refresh token system."
            )

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
        # Determine if this is a refresh token request
        is_refresh = params.get("grant_type") == "refresh_token"

        # Build request body with appropriate grant_type
        request_body = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            **params,
        }

        # Set default grant_type if not provided
        if "grant_type" not in request_body:
            request_body["grant_type"] = "authorization_code"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # For refresh token requests, support HTTP Basic Authentication as recommended
        if is_refresh:
            # Option 1: Use HTTP Basic Auth (preferred by Linear)
            client_credentials = f"{self.client_id}:{self.client_secret}"
            encoded_credentials = base64.b64encode(client_credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_credentials}"

            # Remove client credentials from body when using Basic Auth
            request_body.pop("client_id", None)
            request_body.pop("client_secret", None)

        response = await Requests().post(
            self.token_url, data=request_body, headers=headers
        )

        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get("error", "Unknown error")
                error_description = error_data.get("error_description", "")
                if error_description:
                    error_message = f"{error_message}: {error_description}"
            except json.JSONDecodeError:
                error_message = response.text
            raise LinearAPIException(
                f"Failed to fetch Linear tokens ({response.status}): {error_message}",
                response.status,
            )

        token_data = response.json()

        # Extract token expiration if provided (for new refresh token implementation)
        now = int(time.time())
        expires_in = token_data.get("expires_in")
        access_token_expires_at = None
        if expires_in:
            access_token_expires_at = now + expires_in

        # Get username - preserve from current credentials if refreshing
        username = None
        if current_credentials and is_refresh:
            username = current_credentials.username
        elif "user" in token_data:
            username = token_data["user"].get("name", "Unknown User")
        else:
            # Fetch username using the access token
            username = await self._request_username(token_data["access_token"])

        new_credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=current_credentials.title if current_credentials else None,
            username=username or "Unknown User",
            access_token=token_data["access_token"],
            scopes=(
                token_data["scope"].split(",")
                if "scope" in token_data
                else (current_credentials.scopes if current_credentials else [])
            ),
            refresh_token=token_data.get("refresh_token"),
            access_token_expires_at=access_token_expires_at,
            refresh_token_expires_at=None,  # Linear doesn't provide refresh token expiration
        )

        if current_credentials:
            new_credentials.id = current_credentials.id

        return new_credentials

    async def get_access_token(self, credentials: OAuth2Credentials) -> str:
        """
        Returns a valid access token, handling migration and refresh as needed.

        This overrides the base implementation to handle Linear's token migration
        from old long-lived tokens to new short-lived tokens with refresh tokens.
        """
        # If token has no expiration and no refresh token, it might be an old token
        # that needs migration
        if (
            credentials.access_token_expires_at is None
            and credentials.refresh_token is None
        ):
            try:
                # Attempt to migrate the old token
                migrated_credentials = await self.migrate_old_token(credentials)
                # Update the credentials store would need to be handled by the caller
                # For now, use the migrated credentials for this request
                credentials = migrated_credentials
            except LinearAPIException:
                # Migration failed, try to use the old token as-is
                # This maintains backward compatibility
                pass

        # Use the standard refresh logic from the base class
        if self.needs_refresh(credentials):
            credentials = await self.refresh_tokens(credentials)

        return credentials.access_token.get_secret_value()

    def needs_migration(self, credentials: OAuth2Credentials) -> bool:
        """
        Check if credentials represent an old long-lived token that needs migration.

        Old tokens have no expiration time and no refresh token.
        """
        return (
            credentials.access_token_expires_at is None
            and credentials.refresh_token is None
        )

    async def _request_username(self, access_token: str) -> Optional[str]:
        # Use the LinearClient to fetch user details using GraphQL
        from ._api import LinearClient

        try:
            # Create a temporary OAuth2Credentials object for the LinearClient
            linear_client = LinearClient(
                APIKeyCredentials(
                    api_key=SecretStr(access_token),
                    title="temp",
                    provider=self.PROVIDER_NAME,
                    expires_at=None,
                )
            )  # Temporary credentials for this request

            query = """
                query Viewer {
                    viewer {
                        name
                    }
                }
            """

            response = await linear_client.query(query)
            return response["viewer"]["name"]

        except Exception as e:  # Handle any errors
            print(f"Error fetching username: {e}")
            return None
