from collections.abc import AsyncIterator

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import HTTPException, Security

from backend.data.tenancy import live_resource_permission_barrier


async def require_live_library_create(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    ctx: autogpt_auth_lib.RequestContext = Security(
        autogpt_auth_lib.requires_resource_permission(
            autogpt_auth_lib.OrgAction.CREATE_RESOURCES,
            autogpt_auth_lib.TeamAction.CREATE_AGENTS,
        )
    ),
) -> AsyncIterator[None]:
    async with live_resource_permission_barrier(
        user_id,
        ctx.org_id,
        ctx.team_id,
        autogpt_auth_lib.OrgAction.CREATE_RESOURCES,
        autogpt_auth_lib.TeamAction.CREATE_AGENTS,
    ) as allowed:
        if not allowed:
            raise HTTPException(403, detail="Resource scope is inactive")
        yield


async def require_live_library_delete(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    ctx: autogpt_auth_lib.RequestContext = Security(
        autogpt_auth_lib.requires_resource_permission(
            autogpt_auth_lib.OrgAction.CREATE_RESOURCES,
            autogpt_auth_lib.TeamAction.DELETE_AGENTS,
        )
    ),
) -> AsyncIterator[None]:
    async with live_resource_permission_barrier(
        user_id,
        ctx.org_id,
        ctx.team_id,
        autogpt_auth_lib.OrgAction.CREATE_RESOURCES,
        autogpt_auth_lib.TeamAction.DELETE_AGENTS,
    ) as allowed:
        if not allowed:
            raise HTTPException(403, detail="Resource scope is inactive")
        yield
