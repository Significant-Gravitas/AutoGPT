"""Thin async client for the Bot Connector REST API.

Every outbound call is addressed relative to the ``serviceUrl`` carried on the
inbound activity rather than a fixed base URL, because Microsoft routes each
tenant/region to its own Connector host. That makes ``serviceUrl``
attacker-influenced input which we then attach a bearer token to, so every
request funnels through :func:`auth.is_allowed_service_url` first.
"""

import asyncio
import logging
import time
from typing import Any

import httpx

from . import auth, config

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_SECONDS = 30.0

# Refresh this far ahead of expiry so an in-flight request never races the
# token going stale.
_TOKEN_REFRESH_MARGIN_SECONDS = 300


class TeamsApiError(Exception):
    """A Bot Connector call failed."""


class TeamsClient:
    """Authenticated Bot Connector client. Safe to share across requests."""

    def __init__(self) -> None:
        self._token: str = ""
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()
        # One pooled client for the adapter's lifetime, matching the Telegram
        # adapter — avoids a TCP+TLS handshake per Connector call.
        self._http = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS)

    async def send_activity(
        self, service_url: str, conversation_id: str, activity: dict[str, Any]
    ) -> str | None:
        """Append an activity to a conversation; returns its new activity id."""
        return await self._post_activity(
            service_url,
            f"v3/conversations/{conversation_id}/activities",
            activity,
        )

    async def create_conversation(
        self, service_url: str, payload: dict[str, Any]
    ) -> str | None:
        """Create a conversation (e.g. a 1:1 chat); returns its id.

        Used for proactive delivery, where there is no inbound activity to
        reply to.
        """
        body = await self._request("POST", service_url, "v3/conversations", payload)
        return (body or {}).get("id")

    async def aclose(self) -> None:
        """Release the pooled connections."""
        await self._http.aclose()

    async def get_team_details(
        self, service_url: str, team_id: str
    ) -> dict[str, Any] | None:
        """The team's name and metadata.

        Teams only stamps ``channelData.team.name`` onto install and
        conversation-update activities, not onto ordinary messages — so a
        command handler that needs the name has to ask for it.
        """
        return await self._request("GET", service_url, f"v3/teams/{team_id}")

    async def _post_activity(
        self, service_url: str, path: str, activity: dict[str, Any]
    ) -> str | None:
        body = await self._request("POST", service_url, path, activity)
        return (body or {}).get("id")

    async def _request(
        self,
        method: str,
        service_url: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not auth.is_allowed_service_url(service_url):
            raise TeamsApiError(
                f"refusing to call untrusted serviceUrl {service_url!r}"
            )
        url = f"{service_url.rstrip('/')}/{path}"
        headers = await self.bearer_headers()
        response = await self._http.request(method, url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise TeamsApiError(
                f"{method} {path} failed ({response.status_code}): {response.text[:300]}"
            )
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return None

    async def bearer_headers(self) -> dict[str, str]:
        """Connector auth headers — empty in Playground mode.

        The Agents Playground neither issues nor checks tokens, and there are
        no credentials to mint one from in that mode.
        """
        if config.allow_unverified_requests():
            return {}
        return {"Authorization": f"Bearer {await self._access_token()}"}

    async def _access_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires_at:
            return self._token
        async with self._token_lock:
            if self._token and time.monotonic() < self._token_expires_at:
                return self._token
            self._token, lifetime = await self._mint_access_token()
            self._token_expires_at = time.monotonic() + max(
                lifetime - _TOKEN_REFRESH_MARGIN_SECONDS, 60
            )
        return self._token

    async def _mint_access_token(self) -> tuple[str, float]:
        """Client-credentials grant against the bot's own tenant authority.

        Single-tenant registrations (the only kind Microsoft issues now) must
        use ``login.microsoftonline.com/{tenant}``; the tenant-less
        ``botframework.com`` authority is the multi-tenant form and 401s here.
        """
        url = f"https://login.microsoftonline.com/{config.get_tenant_id()}/oauth2/v2.0/token"
        response = await self._http.post(
            url,
            data={
                "grant_type": "client_credentials",
                "client_id": config.get_app_id(),
                "client_secret": config.get_app_password(),
                "scope": config.CONNECTOR_SCOPE,
            },
        )
        if response.status_code >= 400:
            raise TeamsApiError(
                f"could not mint Connector token ({response.status_code}): "
                f"{response.text[:300]}"
            )
        body = response.json()
        token = body.get("access_token")
        if not token:
            raise TeamsApiError("token response carried no access_token")
        return token, float(body.get("expires_in", 3600))
