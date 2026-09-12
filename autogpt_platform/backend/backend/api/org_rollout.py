"""Gate new explicit collaboration scopes without changing authorization.

The flag is not retroactive data revocation: legacy own-personal org-home
requests keep their normal authorized access, and persisted background execution
keeps its recorded scope. Explicit shared scopes are rejected, never rewritten.
"""

from collections.abc import Callable
from typing import TypeVar

import autogpt_libs.auth as auth
from autogpt_libs.auth.dependencies import (
    get_request_context as resolve_request_context,
)
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from fastapi import FastAPI, HTTPException, Request, Security

from backend.api.features.orgs.db import get_user_default_team
from backend.api.features.orgs.rollout import org_collaboration_enabled

_Endpoint = TypeVar("_Endpoint", bound=Callable[..., object])
_cleanup_endpoints: set[Callable[..., object]] = set()


def install_org_rollout_boundary(app: FastAPI) -> None:
    app.dependency_overrides[auth.get_request_context] = get_rollout_request_context


def org_rollout_cleanup(endpoint: _Endpoint) -> _Endpoint:
    _cleanup_endpoints.add(endpoint)
    return endpoint


async def get_rollout_request_context(
    request: Request,
    jwt_payload: dict = Security(get_jwt_payload),
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
    user_id: str, organization_id: str | None, team_id: str | None
) -> None:
    if await org_collaboration_enabled(user_id):
        return
    personal_org_id, default_team_id = await get_user_default_team(user_id)
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
