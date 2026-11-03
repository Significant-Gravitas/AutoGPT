"""Resolve saved organization identities before crossing memory boundaries."""

from contextlib import asynccontextmanager

from backend.data.db_accessors import experts_db, live_resource_lease, memory_db

from .scope import MemoryScope


async def resolve_memory_scope(
    user_id: str,
    expert_id: str | None = None,
    organization_id: str | None = None,
) -> MemoryScope:
    scope = MemoryScope.build(user_id, expert_id, organization_id)
    if organization_id is None:
        return scope
    personal_org_id = await memory_db().get_personal_memory_organization(user_id)
    if personal_org_id is None:
        raise ValueError("Personal memory organization could not be resolved")
    if organization_id == personal_org_id:
        return MemoryScope.build(user_id, expert_id)
    return scope


async def scope_for_key(
    user_id: str, expert_id: str | None, scope_key: str
) -> MemoryScope:
    """The memory scope of *user_id* (and *expert_id*) whose ``scope_key`` is
    *scope_key*: the owner's personal one, else the scope of the saved
    organization the owner's memory was written in (``MemoryScopeRegistry``).
    A record that names a scope only by its key (a ``DreamPass`` row) is
    rebuilt this way, never as the owner's personal scope. Raises
    ``ValueError`` for a key no scope of the owner has."""
    scope = MemoryScope.build(user_id, expert_id)
    if scope.scope_key == scope_key:
        return scope
    for registered in await memory_db().list_memory_scopes(user_id):
        if registered.scope_key == scope_key and registered.expert_id == expert_id:
            return registered
    raise ValueError("No memory scope of its owner has this key")


async def memory_organization_id(scope: MemoryScope) -> str:
    if scope.organization_id is not None:
        return scope.organization_id
    organization_id = await memory_db().get_personal_memory_organization(
        scope.owner_user_id
    )
    if organization_id is None:
        raise ValueError("Personal memory organization could not be resolved")
    return organization_id


async def owns_memory_scope(scope: MemoryScope) -> bool:
    if scope.expert_id is None:
        return True
    expert = await experts_db().get_expert(
        scope.owner_user_id, scope.expert_id, include_workflows=False
    )
    return expert is not None and scope == await resolve_memory_scope(
        scope.owner_user_id, scope.expert_id, expert.organization_id
    )


@asynccontextmanager
async def memory_scope_access(scope: MemoryScope):
    if not await owns_memory_scope(scope):
        yield None
        return
    organization_id = await memory_organization_id(scope)
    async with live_resource_lease(
        scope.owner_user_id, organization_id, None, "create"
    ) as allowed:
        yield allowed
