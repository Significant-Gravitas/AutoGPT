"""Entra delegated OAuth for Microsoft 365 Copilot.

Worth stating plainly, because it is the exception among the subscription
providers here: this is a *sanctioned* integration. Microsoft documents a
Chat API for third-party applications, its preview terms explicitly
contemplate ISVs and multi-tenant apps, and the flow below is ordinary
authorization-code-with-PKCE against an Entra application the deployment
registers itself. Nothing is impersonated and no undocumented endpoint is
called.

Two consequences of that which shape the code:

- The tenant segment is ``/organizations``, not ``/common``. The Chat API
  is delegated-only and requires a Microsoft 365 Copilot licence, which
  personal Microsoft accounts cannot hold -- so ``/common`` would let
  someone sign in with an account that can never work, and the failure
  would land after the consent screen instead of before it.
- ``offline_access`` is requested explicitly. Without it Entra returns no
  refresh token, and the connection would silently stop working about an
  hour after it was made.
"""

import time
from typing import Optional
from urllib.parse import urlencode

from backend.data.model import OAuth2Credentials
from backend.integrations.providers import ProviderName
from backend.util.request import Requests

from .base import BaseOAuthHandler

_AUTHORITY = "https://login.microsoftonline.com/organizations/oauth2/v2.0"


class Microsoft365CopilotOAuthHandler(BaseOAuthHandler):
    """Docs: https://learn.microsoft.com/en-us/microsoft-365/copilot/extensibility/api/ai-services/chat/overview"""  # noqa: E501

    PROVIDER_NAME = ProviderName.MICROSOFT_365_COPILOT

    # The grounding scopes the Chat API reads over on the user's behalf,
    # plus offline_access for the refresh token. Requested together because
    # Entra's consent screen is per-application, not per-call: asking for
    # them later means a second consent prompt mid-conversation.
    DEFAULT_SCOPES = [
        "Sites.Read.All",
        "Mail.Read",
        "People.Read.All",
        "OnlineMeetingTranscript.Read.All",
        "Chat.Read",
        "ChannelMessage.Read.All",
        "ExternalItem.Read.All",
        "offline_access",
    ]

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_base_url = f"{_AUTHORITY}/authorize"
        self.token_url = f"{_AUTHORITY}/token"

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "response_mode": "query",
            "scope": " ".join(self.handle_default_scopes(scopes)),
            "state": state,
        }
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"
        return f"{self.auth_base_url}?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        body = {
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.handle_default_scopes(scopes)),
        }
        if code_verifier:
            body["code_verifier"] = code_verifier
        return await self._request_tokens(body)

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        """Entra has no token-revocation endpoint for a delegated grant.

        Returning False rather than pretending: the caller uses it to decide
        whether to tell the user the remote side was cleaned up, and saying
        yes here would be a claim we cannot make. Removing our stored copy
        still happens; the user revokes our application from their Microsoft
        account page if they want the grant gone at the source.
        """
        return False

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            return credentials
        return await self._request_tokens(
            {
                "refresh_token": credentials.refresh_token.get_secret_value(),
                "grant_type": "refresh_token",
                "scope": " ".join(credentials.scopes or self.DEFAULT_SCOPES),
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
        response = await Requests().post(
            self.token_url,
            data=request_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token_data: dict = response.json()
        if "access_token" not in token_data:
            raise ValueError(
                token_data.get("error_description")
                or token_data.get("error")
                or "Microsoft did not return an access token"
            )

        now = int(time.time())
        granted = token_data.get("scope", "")
        credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=current_credentials.title if current_credentials else None,
            username=(current_credentials.username if current_credentials else None),
            access_token=token_data["access_token"],
            # Entra rotates the refresh token on every use. Keeping the old
            # one when a response omits it is what stops a rotation from
            # ending the connection.
            refresh_token=(
                token_data.get("refresh_token")
                or (
                    current_credentials.refresh_token.get_secret_value()
                    if current_credentials and current_credentials.refresh_token
                    else None
                )
            ),
            scopes=(
                granted.split()
                if granted
                else (current_credentials.scopes if current_credentials else [])
            ),
            access_token_expires_at=(
                now + int(expires_in)
                if (expires_in := token_data.get("expires_in"))
                else None
            ),
        )
        if current_credentials:
            credentials.id = current_credentials.id
        return credentials
