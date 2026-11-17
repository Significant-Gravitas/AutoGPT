"""Team roles from Microsoft Graph.

The Bot Connector says who sent a message but not their role in the team: its
member records only say user, guest or anonymous. Team ownership lives in
Graph, which the bot reads with its own app credentials. That needs one
application permission, granted either way:

- ``TeamMember.Read.Group``, which ``manifest.json`` requests as
  resource-specific consent, so a team grants it when it installs the app;
- ``TeamMember.Read.All``, granted tenant-wide by an admin.

Without either, Graph answers 403 and callers treat the role as unknown.
"""

import asyncio
import re
import time
from urllib.parse import quote

import httpx

from .api_client import _TOKEN_REFRESH_MARGIN_SECONDS, mint_app_token

GRAPH_SCOPE = "https://graph.microsoft.com/.default"

_GRAPH_URL = "https://graph.microsoft.com/v1.0"
_HTTP_TIMEOUT_SECONDS = 15.0

# Entra object ids are GUIDs. Both ids come from the activity body, so they
# are checked before either goes into a Graph path or $filter expression.
_OBJECT_ID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


class TeamsGraphError(Exception):
    """A Graph lookup could not be made or was refused."""


class TeamsGraphClient:
    """App-only Graph client for team membership reads. Safe to share."""

    def __init__(self) -> None:
        self._token = ""
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()
        self._http = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS)

    async def is_team_owner(self, team_group_id: str, user_object_id: str) -> bool:
        """Whether the Entra user owns the team backed by ``team_group_id``,
        the team's Microsoft 365 group id."""
        if not (
            _OBJECT_ID.fullmatch(team_group_id) and _OBJECT_ID.fullmatch(user_object_id)
        ):
            raise TeamsGraphError("the team or user id is not an Entra object id")
        # Percent-encoded by hand, spaces as %20 as Graph documents: httpx's
        # params= would send them form-style, as "+".
        member_filter = quote(
            "(microsoft.graph.aadUserConversationMember/userId eq "
            f"'{user_object_id}')",
            safe="",
        )
        response = await self._http.get(
            f"{_GRAPH_URL}/teams/{team_group_id}/members?$filter={member_filter}",
            headers={"Authorization": f"Bearer {await self._access_token()}"},
        )
        if response.status_code >= 400:
            raise TeamsGraphError(
                f"team member lookup failed ({response.status_code}): "
                f"{response.text[:300]}"
            )
        members = response.json().get("value") or []
        # Only the user's own record counts, should Graph ever answer with
        # more members than the filter asks for
        return any(
            str(member.get("userId", "")).lower() == user_object_id.lower()
            and "owner" in {str(role).lower() for role in member.get("roles") or []}
            for member in members
        )

    async def aclose(self) -> None:
        """Release the pooled connections."""
        await self._http.aclose()

    async def _access_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires_at:
            return self._token
        async with self._token_lock:
            if self._token and time.monotonic() < self._token_expires_at:
                return self._token
            self._token, lifetime = await mint_app_token(self._http, GRAPH_SCOPE)
            self._token_expires_at = time.monotonic() + max(
                lifetime - _TOKEN_REFRESH_MARGIN_SECONDS, 60
            )
        return self._token
