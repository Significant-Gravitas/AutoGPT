"""Gate new explicit collaboration scopes without changing authorization.

The flag is not retroactive data revocation: own-personal requests without a
team keep their normal authorized access, and persisted background execution
keeps its recorded scope. Explicit shared scopes are rejected, never rewritten.

API keys and OAuth tokens get a stricter gate, ``require_personal_scope``: they
act only in the caller's personal organization and its default team, whatever
the flag says, until credentials get a design for shared organizations.
"""

from collections.abc import Awaitable, Callable
from typing import TypeVar

import autogpt_libs.auth as auth
from autogpt_libs.auth.dependencies import (
    get_request_context as resolve_request_context,
)
from autogpt_libs.auth.dependencies import get_unbanned_jwt_payload
from autogpt_libs.auth.models import RequestContext
from fastapi import FastAPI, HTTPException, Request, Security

from backend.api.features.orgs.db import get_user_default_team
from backend.api.features.orgs.rollout import org_collaboration_enabled

_Endpoint = TypeVar("_Endpoint", bound=Callable[..., object])
_cleanup_endpoints: set[Callable[..., object]] = set()
# The caller's personal org and its default team.
DefaultTeamLookup = Callable[[str], Awaitable[tuple[str | None, str | None]]]


def install_org_rollout_boundary(app: FastAPI) -> None:
    app.dependency_overrides[auth.get_request_context] = get_rollout_request_context


def org_rollout_cleanup(endpoint: _Endpoint) -> _Endpoint:
    _cleanup_endpoints.add(endpoint)
    return endpoint


async def get_rollout_request_context(
    request: Request,
    # The overridden resolver refuses a banned account here; keep that check.
    jwt_payload: dict = Security(get_unbanned_jwt_payload),
) -> RequestContext:
    ctx = await resolve_request_context(request, jwt_payload)
    if request.scope.get("endpoint") in _cleanup_endpoints:
        return ctx
    if any(
        request.headers.get(header, "").strip() for header in ("X-Org-Id", "X-Team-Id")
    ):
        await require_personal_scope_without_collaboration(
            ctx.user_id, ctx.org_id, ctx.team_id
        )
    return ctx


async def require_personal_scope_without_collaboration(
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
    *,
    default_team: DefaultTeamLookup | None = None,
) -> None:
    """``default_team`` replaces the direct database read, for a process
    without Prisma (the websocket server reads through the DatabaseManager)."""
    if await org_collaboration_enabled(user_id):
        return
    personal_org_id, default_team_id = await (default_team or get_user_default_team)(
        user_id
    )
    # Legacy personal requests may omit a team. Never replace an explicitly
    # selected shared scope with personal tenancy: reject it instead.
    if (
        personal_org_id is not None
        and organization_id == personal_org_id
        and (team_id is None or team_id == default_team_id)
    ):
        return
    raise HTTPException(
        status_code=403,
        detail="Organization collaboration is not enabled for this account",
    )


async def require_personal_scope(
    user_id: str, organization_id: str | None, team_id: str | None
) -> tuple[str, str]:
    """The scope an API key or OAuth token acts in: always personal.

    Credentials act only in the caller's own personal organization and its
    default team, whatever the org rollout says, until they get a design for
    shared organizations. A credential naming no organization or no team acts
    there; one naming anything else is refused, never rewritten.
    """
    personal_org_id, default_team_id = await get_user_default_team(user_id)
    if personal_org_id is None or default_team_id is None:
        raise HTTPException(
            status_code=403,
            detail=(
                "No organization context available. Your account may be in an "
                "inconsistent state — please contact support."
            ),
        )
    if organization_id not in (None, personal_org_id) or team_id not in (
        None,
        default_team_id,
    ):
        raise HTTPException(
            status_code=403,
            detail="API access outside your personal organization is not enabled yet",
        )
    return personal_org_id, default_team_id
