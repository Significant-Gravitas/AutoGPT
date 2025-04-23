import json
from typing import Optional
from urllib.parse import urlencode

from pydantic import SecretStr

from backend.blocks.linear._api import LinearAPIException
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.providers import ProviderName
from backend.util.request import requests

from .base import BaseOAuthHandler


class LinearOAuthHandler(BaseOAuthHandler):
    """
    OAuth2 handler for Linear.
    """

    PROVIDER_NAME = ProviderName.LINEAR

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_base_url = "https://linear.app/oauth/authorize"
        self.token_url = "https://api.linear.app/oauth/token"  # Correct token URL
        self.revoke_url = "https://api.linear.app/oauth/revoke"

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

    def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        return self._request_tokens({"code": code, "redirect_uri": self.redirect_uri})

    def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        if not credentials.access_token:
            raise ValueError("No access token to revoke")

        headers = {
            "Authorization": f"Bearer {credentials.access_token.get_secret_value()}"
        }

        response = requests.post(self.revoke_url, headers=headers)
        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get("error", "Unknown error")
            except json.JSONDecodeError:
                error_message = response.text
            raise LinearAPIException(
                f"Failed to revoke Linear tokens ({response.status_code}): {error_message}",
                response.status_code,
            )

        return True  # Linear doesn't return JSON on successful revoke

    def _refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
        if not credentials.refresh_token:
            raise ValueError(
                "No refresh token available."
            )  # Linear uses non-expiring tokens

        return self._request_tokens(
            {
                "refresh_token": credentials.refresh_token.get_secret_value(),
                "grant_type": "refresh_token",
            }
        )

    def _request_tokens(
        self,
        params: dict[str, str],
        current_credentials: Optional[OAuth2Credentials] = None,
    ) -> OAuth2Credentials:
        request_body = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "authorization_code",  # Ensure grant_type is correct
            **params,
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }  # Correct header for token request
        response = requests.post(self.token_url, data=request_body, headers=headers)

        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get("error", "Unknown error")

            except json.JSONDecodeError:
                error_message = response.text
            raise LinearAPIException(
                f"Failed to fetch Linear tokens ({response.status_code}): {error_message}",
                response.status_code,
            )

        token_data = response.json()

        # Note: Linear access tokens do not expire, so we set expires_at to None
        new_credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=current_credentials.title if current_credentials else None,
            username=token_data.get("user", {}).get(
                "name", "Unknown User"
            ),  # extract name or set appropriate
            access_token=token_data["access_token"],
            scopes=token_data["scope"].split(
                ","
            ),  # Linear returns comma-separated scopes
            refresh_token=token_data.get(
                "refresh_token"
            ),  # Linear uses non-expiring tokens so this might be null
            access_token_expires_at=None,
            refresh_token_expires_at=None,
        )
        if current_credentials:
            new_credentials.id = current_credentials.id
        return new_credentials

    def _request_username(self, access_token: str) -> Optional[str]:

        # Use the LinearClient to fetch user details using GraphQL
        from backend.blocks.linear._api import LinearClient

        try:

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

            response = linear_client.query(query)
            return response["viewer"]["name"]

        except Exception as e:  # Handle any errors

            print(f"Error fetching username: {e}")
            return None
