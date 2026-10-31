"""
Guard for Exa calls that must not run on the platform's shared Exa key.

Every AutoGPT user who picks the platform-provided Exa credential shares one Exa
team. Most Exa calls are scoped to the request (a search, a run fetched by its
ID), but some act on the whole team: listing runs or monitors, team-wide
webhooks and events, batch actions, and monitors that keep running searches on
Exa's schedule with no cost the platform can bill. On the shared key those would
show, change or spend on other users' resources, so they need the user's own key.
"""

from typing import Any

from backend.integrations.credentials_store import is_system_credential
from backend.sdk import APIKeyCredentials

from ._exa_rest import QueryParams, exa_request

PLATFORM_KEY_ID = "exa-default"


def is_platform_exa_key(credentials: APIKeyCredentials) -> bool:
    """Whether ``credentials`` is an Exa key the platform provides to everyone."""
    return credentials.id == PLATFORM_KEY_ID or is_system_credential(credentials.id)


def require_own_exa_key(credentials: APIKeyCredentials, feature: str) -> None:
    """Refuse a team-wide Exa operation on the shared platform key."""
    if not is_platform_exa_key(credentials):
        return
    raise ValueError(
        f"{feature} needs your own Exa API key. The platform's Exa key is shared "
        "by every user, and this acts on everything in that shared Exa account. "
        "Add your key from dashboard.exa.ai under Integrations, then select it "
        "on this block."
    )


class TeamScopedExaMixin:
    """``call_exa`` for blocks whose Exa resources belong to the whole team.

    Same mockable entry point as ``ExaRestMixin``, but it refuses the shared
    platform key before any request is sent.
    """

    TEAM_SCOPED_FEATURE = "This Exa block"

    @classmethod
    async def call_exa(
        cls,
        credentials: APIKeyCredentials,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
        params: QueryParams | None = None,
    ) -> dict[str, Any]:
        require_own_exa_key(credentials, cls.TEAM_SCOPED_FEATURE)
        return await exa_request(credentials, method, url, body=body, params=params)
