"""User-delegated OAuth for GitHub Copilot.

Links a user's GitHub account so their chats run on -- and bill to -- their
own Copilot subscription.

Three things here look like mistakes and are not:

**Registered as a GitHub App, not a classic OAuth App.** Both work, and
GitHub says so. The difference that matters for a stored third-party
credential is expiry: a GitHub App user token (``ghu_``) lasts 8 hours and
comes with a 6-month refresh token, while an OAuth App token (``gho_``) has
no documented expiry and no refresh token at all. A credential that never
rotates is one that stays valid for as long as it stays leaked, so the
rotating one is the right default. GitHub Apps can have expiry turned off in
their settings -- for this integration, leave it on.

**No Copilot scope is requested, because none exists.** Copilot access is
gated by whether the user holds a seat, not by an OAuth scope. ``read:user``
is what GitHub's own sample and Zed both send, and it is what identifies the
account for linking. (``copilot_requests`` is a fine-grained *PAT*
permission, not something an OAuth flow can ask for -- a natural thing to go
looking for and a dead end.)

**A classic PAT is silently ignored downstream.** ``ghp_`` tokens reach the
Copilot runtime and do nothing, with no error to explain it, so this handler
only ever produces the token types that work.

Not implemented here: reaching the model. Unlike the OpenAI-compatible
providers there is no documented chat-completions endpoint -- the sanctioned
path is a ``copilot`` runtime process spoken to over JSON-RPC, so that lives
in its own module rather than behind ``SubscriptionEndpoint``.
"""

import time
from typing import Optional
from urllib.parse import urlencode

from backend.data.model import OAuth2Credentials
from backend.integrations.providers import ProviderName
from backend.util.request import Requests

from .base import BaseOAuthHandler


class GitHubCopilotOAuthHandler(BaseOAuthHandler):
    """Docs: https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/refreshing-user-access-tokens"""  # noqa: E501

    PROVIDER_NAME = ProviderName.GITHUB_COPILOT

    # There is no `copilot` scope. This one names the account for linking and
    # is what GitHub's own sample sends; a seat, not a scope, is what grants
    # Copilot access.
    DEFAULT_SCOPES = ["read:user"]

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_base_url = "https://github.com/login/oauth/authorize"
        self.token_url = "https://github.com/login/oauth/access_token"

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.handle_default_scopes(scopes)),
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

        await Requests().delete(
            url=f"https://api.github.com/applications/{self.client_id}/token",
            auth=(self.client_id, self.client_secret),
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"access_token": credentials.access_token.get_secret_value()},
        )
        return True

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            # An OAuth App token, or a GitHub App with expiry turned off.
            # Nothing to refresh, and it does not expire -- returning it
            # unchanged is correct rather than a silent failure.
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
        response = await Requests().post(
            self.token_url,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                **params,
            },
            headers={"Accept": "application/json"},
        )
        token_data: dict = response.json()
        if "access_token" not in token_data:
            # GitHub answers 200 with an error body rather than a 4xx here,
            # so a caller that only checks the status sees a success and
            # stores a credential with no token in it.
            raise ValueError(
                token_data.get("error_description")
                or token_data.get("error")
                or "GitHub did not return an access token"
            )

        access_token = token_data["access_token"]
        if access_token.startswith("ghp_"):
            # Accepted by GitHub, then ignored by the Copilot runtime with no
            # error. Refusing here means the failure lands at sign-in with a
            # reason, instead of on every chat with none.
            raise ValueError(
                "A classic personal access token cannot be used for Copilot. "
                "Sign in through the app instead."
            )

        now = int(time.time())
        username = current_credentials.username if current_credentials else None
        if username is None:
            username = await self._request_username(access_token)

        credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=current_credentials.title if current_credentials else None,
            username=username,
            access_token=access_token,
            refresh_token=token_data.get("refresh_token"),
            scopes=(
                token_data.get("scope", "").split(",")
                or (current_credentials.scopes if current_credentials else [])
            ),
            access_token_expires_at=(
                now + int(expires_in)
                if (expires_in := token_data.get("expires_in"))
                else None
            ),
            refresh_token_expires_at=(
                now + int(expires_in)
                if (expires_in := token_data.get("refresh_token_expires_in"))
                else None
            ),
        )
        if current_credentials:
            credentials.id = current_credentials.id
        return credentials

    async def _request_username(self, access_token: str) -> str | None:
        """The GitHub login, so the connection can be named on screen.

        Best-effort: a connection that works but cannot say whose it is beats
        a sign-in that fails because a display detail could not be fetched.
        """
        try:
            response = await Requests().get(
                "https://api.github.com/user",
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {access_token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            return response.json().get("login")
        except Exception:
            return None
