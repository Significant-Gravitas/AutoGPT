"""
Slack OAuth handler implementation.

Uses Slack's bot-install OAuth v2 flow (``oauth/v2/authorize`` +
``oauth.v2.access``), not "Sign in with Slack" (OpenID Connect) — the OIDC
flow only yields an identity token and cannot call Slack Web API methods
like ``chat.postMessage``.

Docs: https://docs.slack.dev/authentication/installing-with-oauth
"""

import logging
import time
from typing import Any, Optional
from urllib.parse import urlencode

from backend.sdk import BaseOAuthHandler, OAuth2Credentials, ProviderName, Requests

logger = logging.getLogger(__name__)

SLACK_AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
SLACK_TOKEN_URL = "https://slack.com/api/oauth.v2.access"
SLACK_REVOKE_URL = "https://slack.com/api/auth.revoke"


class SlackOAuthError(Exception):
    """Raised when Slack's Web API returns ``ok: false`` (Slack signals
    OAuth failures this way even on an HTTP 200 response)."""


class SlackOAuthHandler(BaseOAuthHandler):
    """
    OAuth2 handler for Slack's bot-install flow.

    Classic Slack bot tokens (``xoxb-...``) never expire and have no refresh
    token. Only workspaces where the installing app has opted into "token
    rotation" issue expiring tokens with a ``refresh_token`` — this handler
    supports both by treating refresh as a no-op when there's nothing to
    refresh. Slack's OAuth v2 endpoint does not support PKCE.
    """

    PROVIDER_NAME = ProviderName.SLACK
    DEFAULT_SCOPES = ["chat:write"]

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        scopes = self.handle_default_scopes(scopes)

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": ",".join(scopes),
            "state": state,
        }
        # Slack's OAuth v2 endpoint does not support PKCE; code_challenge is
        # accepted by the interface but has nothing to attach to here.
        return f"{SLACK_AUTHORIZE_URL}?{urlencode(params)}"

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

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        # Classic (non-rotating) Slack bot tokens never expire and are never
        # issued with a refresh token. Only apps with "token rotation"
        # enabled produce one — if there isn't one, there's nothing to do.
        if credentials.refresh_token is None:
            logger.debug("Slack credentials have no refresh token; skipping refresh")
            return credentials

        return await self._request_tokens(
            {
                "grant_type": "refresh_token",
                "refresh_token": credentials.refresh_token.get_secret_value(),
            },
            current_credentials=credentials,
        )

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        if not credentials.access_token:
            raise ValueError("No access token to revoke")

        response = await Requests().post(
            SLACK_REVOKE_URL,
            headers={
                "Authorization": f"Bearer {credentials.access_token.get_secret_value()}"
            },
        )
        result: dict[str, Any] = response.json()
        return bool(result.get("ok"))

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
            SLACK_TOKEN_URL,
            data=request_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token_data: dict[str, Any] = response.json()

        # Slack returns HTTP 200 even on failure; success is only signalled
        # by "ok": true. Never assume a 200 response is a valid token.
        if not token_data.get("ok"):
            raise SlackOAuthError(token_data.get("error", "unknown_error"))

        team = token_data.get("team") or {}
        metadata = {
            **(current_credentials.metadata if current_credentials else {}),
            "team_id": team.get("id"),
            "team_name": team.get("name"),
            "bot_user_id": token_data.get("bot_user_id"),
            "app_id": token_data.get("app_id"),
        }

        scope_str = token_data.get("scope")
        new_credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=current_credentials.title if current_credentials else None,
            username=team.get("name"),
            access_token=token_data["access_token"],
            access_token_expires_at=(
                int(time.time()) + expires_in
                if (expires_in := token_data.get("expires_in"))
                else None
            ),
            refresh_token=token_data.get("refresh_token"),
            refresh_token_expires_at=None,
            scopes=(
                scope_str.split(",")
                if scope_str
                else (current_credentials.scopes if current_credentials else [])
            ),
            metadata=metadata,
        )

        if current_credentials:
            new_credentials.id = current_credentials.id

        return new_credentials
