from collections.abc import AsyncIterator, Callable
from typing import TypeVar, cast

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import Depends, HTTPException, Security
from fastapi.params import Depends as DependsParameter

from backend.data.tenancy import (
    live_actor_org_permission_barrier,
    live_org_permission_barrier,
    live_resource_permission_barrier,
)

_T = TypeVar("_T")


def live_dependency(
    dependency: Callable[..., AsyncIterator[_T]],
) -> DependsParameter:
    return Depends(dependency, scope="function")


def requires_live_resource_permission(
    org_action: autogpt_auth_lib.OrgAction,
    team_action: autogpt_auth_lib.TeamAction,
) -> autogpt_auth_lib.RequestContext:
    async def dependency(
        user_id: str = Security(autogpt_auth_lib.get_user_id),
        ctx: autogpt_auth_lib.RequestContext = Security(
            autogpt_auth_lib.requires_resource_permission(org_action, team_action)
        ),
    ) -> AsyncIterator[autogpt_auth_lib.RequestContext]:
        async with live_resource_permission_barrier(
            user_id,
            ctx.org_id,
            ctx.team_id,
            org_action,
            team_action,
        ) as allowed:
            if not allowed:
                raise HTTPException(403, detail="Resource scope is inactive")
            yield ctx

    return cast(autogpt_auth_lib.RequestContext, live_dependency(dependency))


def requires_live_org_permission(
    action: autogpt_auth_lib.OrgAction,
) -> autogpt_auth_lib.RequestContext:
    async def dependency(
        ctx: autogpt_auth_lib.RequestContext = Security(
            autogpt_auth_lib.requires_org_permission(action)
        ),
    ) -> AsyncIterator[autogpt_auth_lib.RequestContext]:
        if ctx.org_id is None:
            raise HTTPException(403, detail="Organization access is inactive")
        async with live_org_permission_barrier(
            ctx.user_id, ctx.org_id, action
        ) as allowed:
            if not allowed:
                raise HTTPException(403, detail="Organization access is inactive")
            yield ctx

    return cast(autogpt_auth_lib.RequestContext, live_dependency(dependency))


def requires_live_actor_org_permission(
    action: autogpt_auth_lib.OrgAction,
) -> autogpt_auth_lib.RequestContext:
    async def dependency(
        ctx: autogpt_auth_lib.RequestContext = Security(
            autogpt_auth_lib.requires_org_permission(action)
        ),
    ) -> AsyncIterator[autogpt_auth_lib.RequestContext]:
        if ctx.org_id is None:
            raise HTTPException(403, detail="Organization access is inactive")
        async with live_actor_org_permission_barrier(
            ctx.user_id, ctx.org_id, action
        ) as allowed:
            if not allowed:
                raise HTTPException(403, detail="Organization access is inactive")
            yield ctx

    return cast(autogpt_auth_lib.RequestContext, live_dependency(dependency))
