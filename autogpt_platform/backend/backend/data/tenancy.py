"""Shared org/team visibility predicate for tenancy-scoped reads.

One definition of "what can this user see in this org" so every list
and fetch surface applies identical semantics:

- org-home rows (``teamId`` NULL) are visible to every org member
- team rows are visible to members of that team
- a user's own rows are always visible to them within the org
- untagged rows (created before org tagging, not yet backfilled) stay
  visible to their owning user

``organization_id`` must come from a membership-verified RequestContext
(or an equally trusted source such as ExecutionContext) — this module
does not re-verify org membership.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from contextvars import Context, ContextVar, copy_context
from datetime import timedelta
from typing import AsyncIterator, Literal
from uuid import uuid4

from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import (
    OrgAction,
    TeamAction,
    check_org_permission,
    check_team_permission,
)
from prisma import Prisma

from backend.data.db import CONN_LIMIT, DATABASE_URL, HTTP_TIMEOUT, add_param, prisma
from backend.util.retry import conn_retry

logger = logging.getLogger(__name__)
LIVE_ACTION_TRANSACTION_TIMEOUT = timedelta(minutes=31)
LIVE_REQUEST_TRANSACTION_LIMIT = max(1, min(2, int(CONN_LIMIT or 5) - 1))
LIVE_TRANSACTION_LEASE_CLASS_LIMIT = 3
LIVE_TRANSACTION_LEASE_POOL_LIMIT = LIVE_TRANSACTION_LEASE_CLASS_LIMIT * 3 + 3
_live_request_transaction_slots: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}
LiveTransactionLeaseClass = Literal["resource", "attachment", "delivery"]
_live_transaction_lease_slots: dict[
    asyncio.AbstractEventLoop, dict[LiveTransactionLeaseClass, asyncio.Semaphore]
] = {}
_active_live_request_transaction: ContextVar[tuple[object, Prisma, Prisma] | None] = (
    ContextVar("active_live_request_transaction", default=None)
)
live_transaction_lease_prisma = Prisma(
    auto_register=False,
    http={"timeout": HTTP_TIMEOUT},
    datasource={
        "url": add_param(
            DATABASE_URL,
            "connection_limit",
            str(LIVE_TRANSACTION_LEASE_POOL_LIMIT),
        )
    },
)


def _get_live_request_transaction_slots() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    slots = _live_request_transaction_slots.get(loop)
    if slots is None:
        slots = asyncio.Semaphore(LIVE_REQUEST_TRANSACTION_LIMIT)
        _live_request_transaction_slots[loop] = slots
    return slots


def _get_live_transaction_lease_slots(
    lease_class: LiveTransactionLeaseClass,
) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    slots = _live_transaction_lease_slots.get(loop)
    if slots is None:
        new_slots: dict[LiveTransactionLeaseClass, asyncio.Semaphore] = {
            name: asyncio.Semaphore(LIVE_TRANSACTION_LEASE_CLASS_LIMIT)
            for name in ("resource", "attachment", "delivery")
        }
        _live_transaction_lease_slots[loop] = new_slots
        slots = new_slots
    return slots[lease_class]


@conn_retry("Prisma lease", "Acquiring connection")
async def connect_live_transaction_lease_database() -> bool:
    if live_transaction_lease_prisma.is_connected():
        return False
    await live_transaction_lease_prisma.connect()
    return True


@conn_retry("Prisma lease", "Releasing connection")
async def disconnect_live_transaction_lease_database() -> None:
    if live_transaction_lease_prisma.is_connected():
        await live_transaction_lease_prisma.disconnect()


def is_live_transaction_lease_database_connected() -> bool:
    return live_transaction_lease_prisma.is_connected()


async def check_live_transaction_lease_database() -> bool:
    if not live_transaction_lease_prisma.is_connected():
        return False
    result = await live_transaction_lease_prisma.query_raw("SELECT 1 AS health_check")
    return result == [{"health_check": 1}]


@asynccontextmanager
async def _live_transaction_lease_slot(
    lease_class: LiveTransactionLeaseClass,
) -> AsyncIterator[Prisma]:
    async with _get_live_transaction_lease_slots(lease_class):
        yield live_transaction_lease_prisma


@asynccontextmanager
async def live_request_transaction(
    client: Prisma | None = None,
) -> AsyncIterator[Prisma]:
    task = asyncio.current_task()
    request_client = client or prisma
    active = _active_live_request_transaction.get()
    if active is not None and active[0] is task and active[1] is request_client:
        yield active[2]
        return

    async with _get_live_request_transaction_slots():
        async with request_client.tx(timeout=LIVE_ACTION_TRANSACTION_TIMEOUT) as tx:
            token = _active_live_request_transaction.set((task, request_client, tx))
            try:
                yield tx
            finally:
                _active_live_request_transaction.reset(token)


@asynccontextmanager
async def _live_context_transaction(
    request_limited: bool, client: Prisma | None = None
) -> AsyncIterator[Prisma]:
    request_client = client or prisma
    if request_limited:
        async with live_request_transaction(request_client) as tx:
            yield tx
        return
    async with request_client.tx(timeout=LIVE_ACTION_TRANSACTION_TIMEOUT) as tx:
        yield tx


@asynccontextmanager
async def _request_or_direct_transaction(
    client: Prisma | None = None,
) -> AsyncIterator[Prisma]:
    request_client = client or prisma
    task = asyncio.current_task()
    active = _active_live_request_transaction.get()
    if active is not None and active[0] is task and active[1] is request_client:
        yield active[2]
        return
    async with request_client.tx(timeout=LIVE_ACTION_TRANSACTION_TIMEOUT) as tx:
        yield tx


async def get_user_team_ids(user_id: str, organization_id: str) -> list[str]:
    """Resource-capable ACTIVE teams for a user in an organization."""
    org_member = await prisma.orgmember.find_first(
        where={
            "userId": user_id,
            "orgId": organization_id,
            "status": "ACTIVE",
            "Org": {"is": {"deletedAt": None}},
        }
    )
    if org_member is None or not (
        org_member.isOwner or org_member.isAdmin or not org_member.isBillingManager
    ):
        return []

    memberships = await prisma.teammember.find_many(
        where={
            "userId": user_id,
            "status": "ACTIVE",
            "Team": {"is": {"orgId": organization_id, "archivedAt": None}},
        }
    )
    return [
        membership.teamId
        for membership in memberships
        if membership.isAdmin or not membership.isBillingManager
    ]


async def has_live_tenancy(
    user_id: str,
    organization_id: str | None,
    team_id: str | None = None,
) -> bool:
    if not organization_id:
        return True

    return await _get_live_context(user_id, organization_id, team_id) is not None


ResourceAccess = Literal["view", "create", "execute", "delete"]

_active_live_scopes: ContextVar[frozenset[tuple[str, str, str | None]]] = ContextVar(
    "active_live_tenancy_scopes", default=frozenset()
)
_active_actor_scopes: ContextVar[frozenset[tuple[str, str]]] = ContextVar(
    "active_actor_tenancy_scopes", default=frozenset()
)
_active_graph_scopes: ContextVar[frozenset[str]] = ContextVar(
    "active_graph_scopes", default=frozenset()
)
_live_resource_leases: dict[str, tuple[asyncio.Event, asyncio.Task[None]]] = {}
_graph_attachment_leases: dict[str, tuple[asyncio.Event, asyncio.Task[None]]] = {}
_alert_condition_leases: dict[str, tuple[asyncio.Event, asyncio.Task[None]]] = {}
_store_listing_version_leases: dict[str, tuple[asyncio.Event, asyncio.Task[None]]] = {}
_live_transaction_lease_holders: dict[asyncio.Task[None], asyncio.Event] = {}


def _track_live_transaction_lease_task(
    task: asyncio.Task[None], release: asyncio.Event, ready: asyncio.Future[object]
) -> asyncio.Task[None]:
    _live_transaction_lease_holders[task] = release

    def finish(completed: asyncio.Task[None]) -> None:
        if completed.cancelled() and not ready.done():
            ready.cancel()
        _live_transaction_lease_holders.pop(completed, None)

    task.add_done_callback(finish)
    return task


async def _cancel_live_transaction_lease_acquisition(
    task: asyncio.Task[None], release: asyncio.Event, ready: asyncio.Future[object]
) -> None:
    release.set()
    if not ready.done():
        ready.cancel()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def release_all_live_transaction_leases() -> None:
    while _live_transaction_lease_holders:
        holders = list(_live_transaction_lease_holders.items())
        for task, release in holders:
            release.set()
            task.cancel()
        await asyncio.gather(
            *(task for task, _ in holders),
            return_exceptions=True,
        )
        await asyncio.sleep(0)


def context_without_live_tenancy_scopes() -> Context:
    context = copy_context()

    def clear() -> None:
        _active_live_scopes.set(frozenset())
        _active_actor_scopes.set(frozenset())
        _active_graph_scopes.set(frozenset())

    context.run(clear)
    return context


async def has_live_resource_access(
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
    access: ResourceAccess,
) -> bool:
    if not organization_id:
        return True

    ctx = await _get_live_context(user_id, organization_id, team_id)
    if ctx is None:
        return False

    org_action, team_action = {
        "view": (OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
        "create": (OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS),
        "execute": (OrgAction.EXECUTE_RESOURCES, TeamAction.EXECUTE_AGENTS),
        "delete": (OrgAction.CREATE_RESOURCES, TeamAction.DELETE_AGENTS),
    }[access]
    if not check_org_permission(ctx, org_action):
        return False
    return team_id is None or check_team_permission(ctx, team_action)


async def has_live_resource_permission(
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
    org_action: OrgAction,
    team_action: TeamAction,
) -> bool:
    if not organization_id:
        return True
    ctx = await _get_live_context(user_id, organization_id, team_id)
    return (
        ctx is not None
        and check_org_permission(ctx, org_action)
        and (team_id is None or check_team_permission(ctx, team_action))
    )


async def _get_live_context(
    user_id: str,
    organization_id: str,
    team_id: str | None,
    client: Prisma | None = None,
) -> RequestContext | None:
    db = client or prisma
    org_member = await db.orgmember.find_first(
        where={
            "userId": user_id,
            "orgId": organization_id,
            "status": "ACTIVE",
            "Org": {"is": {"deletedAt": None}},
        }
    )
    if org_member is None:
        return None

    team_member = (
        await db.teammember.find_first(
            where={
                "userId": user_id,
                "teamId": team_id,
                "status": "ACTIVE",
                "Team": {"is": {"orgId": organization_id, "archivedAt": None}},
            }
        )
        if team_id is not None
        else None
    )
    if team_id is not None and team_member is None:
        return None

    return RequestContext(
        user_id=user_id,
        org_id=organization_id,
        team_id=team_id,
        is_org_owner=org_member.isOwner,
        is_org_admin=org_member.isAdmin,
        is_org_billing_manager=org_member.isBillingManager,
        is_team_admin=bool(team_member and team_member.isAdmin),
        is_team_billing_manager=bool(team_member and team_member.isBillingManager),
        seat_status="ACTIVE",
    )


async def _lock_live_scope(
    client: Prisma,
    user_id: str,
    organization_id: str,
    team_id: str | None,
    *,
    actor_scope_locked: bool = False,
) -> None:
    if not actor_scope_locked:
        await _lock_advisory_scope(
            client,
            f"tenancy:org-user:{organization_id}:{user_id}",
            shared=True,
        )
    await _lock_advisory_scope(client, f"tenancy:org:{organization_id}", shared=True)
    if team_id is not None:
        await _lock_advisory_scope(client, f"tenancy:team:{team_id}", shared=True)


async def _lock_advisory_scope(
    client: Prisma, scope: str, *, shared: bool = False
) -> None:
    if shared:
        await client.execute_raw(
            "SELECT pg_advisory_xact_lock_shared(hashtextextended($1, 0))",
            scope,
        )
    else:
        await client.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            scope,
        )


async def lock_live_org_scope(client: Prisma, organization_id: str) -> None:
    await _lock_advisory_scope(client, f"tenancy:org:{organization_id}")


async def lock_live_org_membership_scope(
    client: Prisma, organization_id: str, user_id: str
) -> None:
    await lock_live_org_membership_scopes(client, organization_id, [user_id])


async def lock_live_org_membership_scopes(
    client: Prisma, organization_id: str, user_ids: list[str]
) -> None:
    for user_id in sorted(set(user_ids)):
        await _lock_advisory_scope(
            client, f"tenancy:org-user:{organization_id}:{user_id}"
        )
    await lock_live_org_scope(client, organization_id)


async def lock_live_team_scope(client: Prisma, team_id: str) -> None:
    await _lock_advisory_scope(client, f"tenancy:team:{team_id}")


async def lock_live_org_permission_scope(
    client: Prisma,
    user_id: str,
    organization_id: str,
    action: OrgAction,
    related_user_ids: list[str] | None = None,
) -> RequestContext | None:
    await lock_live_org_membership_scopes(
        client,
        organization_id,
        [user_id, *(related_user_ids or [])],
    )
    ctx = await _get_live_context(user_id, organization_id, None, client)
    return ctx if ctx is not None and check_org_permission(ctx, action) else None


async def lock_live_org_or_team_permission_scope(
    client: Prisma,
    user_id: str,
    organization_id: str,
    team_id: str,
    org_action: OrgAction,
    team_action: TeamAction,
    related_user_ids: list[str] | None = None,
) -> RequestContext | None:
    await lock_live_org_membership_scopes(
        client,
        organization_id,
        [user_id, *(related_user_ids or [])],
    )
    await lock_live_team_scope(client, team_id)
    org_ctx = await _get_live_context(user_id, organization_id, None, client)
    if org_ctx is None:
        return None
    if check_org_permission(org_ctx, org_action):
        return org_ctx
    team_ctx = await _get_live_context(user_id, organization_id, team_id, client)
    return (
        team_ctx
        if team_ctx is not None and check_team_permission(team_ctx, team_action)
        else None
    )


@asynccontextmanager
async def _live_context_barrier(
    user_id: str,
    organization_id: str,
    team_id: str | None,
    request_limited: bool = True,
    client: Prisma | None = None,
) -> AsyncIterator[RequestContext | None]:
    scope = (user_id, organization_id, team_id)
    actor_scope = (user_id, organization_id)
    active_scopes = _active_live_scopes.get()
    active_actor_scopes = _active_actor_scopes.get()
    if scope in active_scopes:
        yield await _get_live_context(user_id, organization_id, team_id)
        return

    async with _live_context_transaction(request_limited, client) as tx:
        await _lock_live_scope(
            tx,
            user_id,
            organization_id,
            team_id,
            actor_scope_locked=actor_scope in active_actor_scopes,
        )
        ctx = await _get_live_context(user_id, organization_id, team_id, tx)
        live_token = _active_live_scopes.set(active_scopes | {scope})
        actor_token = None
        if actor_scope not in active_actor_scopes:
            actor_token = _active_actor_scopes.set(active_actor_scopes | {actor_scope})
        try:
            yield ctx
        finally:
            if actor_token is not None:
                _active_actor_scopes.reset(actor_token)
            _active_live_scopes.reset(live_token)


@asynccontextmanager
async def _live_actor_context(
    user_id: str, organization_id: str
) -> AsyncIterator[Prisma]:
    scope = (user_id, organization_id)
    active_scopes = _active_actor_scopes.get()
    if scope in active_scopes:
        yield prisma
        return

    async with live_request_transaction() as tx:
        await _lock_advisory_scope(
            tx,
            f"tenancy:org-user:{organization_id}:{user_id}",
            shared=True,
        )
        token = _active_actor_scopes.set(active_scopes | {scope})
        try:
            yield tx
        finally:
            _active_actor_scopes.reset(token)


def _allows_resource_access(
    ctx: RequestContext | None,
    team_id: str | None,
    access: ResourceAccess,
) -> bool:
    if ctx is None:
        return False
    org_action, team_action = {
        "view": (OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
        "create": (OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS),
        "execute": (OrgAction.EXECUTE_RESOURCES, TeamAction.EXECUTE_AGENTS),
        "delete": (OrgAction.CREATE_RESOURCES, TeamAction.DELETE_AGENTS),
    }[access]
    return check_org_permission(ctx, org_action) and (
        team_id is None or check_team_permission(ctx, team_action)
    )


@asynccontextmanager
async def live_resource_access_barrier(
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
    access: ResourceAccess,
    request_limited: bool = True,
    client: Prisma | None = None,
) -> AsyncIterator[bool]:
    if organization_id is None:
        yield True
        return
    async with _live_context_barrier(
        user_id,
        organization_id,
        team_id,
        request_limited=request_limited,
        client=client,
    ) as ctx:
        yield _allows_resource_access(ctx, team_id, access)


@asynccontextmanager
async def _live_resource_lease_barrier(
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
    access: ResourceAccess,
) -> AsyncIterator[bool]:
    if organization_id is None:
        async with live_resource_access_barrier(
            user_id,
            organization_id,
            team_id,
            access,
            request_limited=False,
        ) as allowed:
            yield allowed
        return

    async with _live_transaction_lease_slot("resource") as client:
        async with live_resource_access_barrier(
            user_id,
            organization_id,
            team_id,
            access,
            request_limited=False,
            client=client,
        ) as allowed:
            yield allowed


async def acquire_live_resource_lease(
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
    access: ResourceAccess,
) -> str | None:
    release = asyncio.Event()
    ready = asyncio.get_running_loop().create_future()
    lease_id = str(uuid4())

    async def hold() -> None:
        try:
            async with _live_resource_lease_barrier(
                user_id, organization_id, team_id, access
            ) as allowed:
                if not ready.done():
                    ready.set_result(allowed)
                if allowed:
                    await release.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            else:
                logger.exception("Live resource lease %s failed", lease_id)
        finally:
            _live_resource_leases.pop(lease_id, None)

    task = _track_live_transaction_lease_task(
        asyncio.create_task(hold(), name=f"live-resource-lease:{lease_id}"),
        release,
        ready,
    )
    try:
        allowed = await asyncio.shield(ready)
    except BaseException:
        await _cancel_live_transaction_lease_acquisition(task, release, ready)
        raise
    if not allowed:
        await task
        return None
    _live_resource_leases[lease_id] = (release, task)
    return lease_id


async def acquire_live_resource_scopes_lease(
    user_id: str,
    scopes: list[tuple[str, str | None]],
    access: ResourceAccess,
) -> tuple[str, list[tuple[str, str | None]]]:
    normalized = sorted(set(scopes), key=lambda scope: (scope[0], scope[1] or ""))
    release = asyncio.Event()
    ready = asyncio.get_running_loop().create_future()
    lease_id = str(uuid4())

    async def hold() -> None:
        try:
            async with _live_transaction_lease_slot("resource") as client:
                async with client.tx(timeout=LIVE_ACTION_TRANSACTION_TIMEOUT) as tx:
                    locked_actors: set[tuple[str, str]] = set()
                    for organization_id, team_id in normalized:
                        actor_scope = (user_id, organization_id)
                        await _lock_live_scope(
                            tx,
                            user_id,
                            organization_id,
                            team_id,
                            actor_scope_locked=actor_scope in locked_actors,
                        )
                        locked_actors.add(actor_scope)

                    authorized: list[tuple[str, str | None]] = []
                    for organization_id, team_id in normalized:
                        ctx = await _get_live_context(
                            user_id, organization_id, team_id, tx
                        )
                        if _allows_resource_access(ctx, team_id, access):
                            authorized.append((organization_id, team_id))

                    if not ready.done():
                        ready.set_result(authorized)
                    await release.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            else:
                logger.exception("Multi-scope live resource lease %s failed", lease_id)
        finally:
            _live_resource_leases.pop(lease_id, None)

    task = _track_live_transaction_lease_task(
        asyncio.create_task(hold(), name=f"live-resource-scopes-lease:{lease_id}"),
        release,
        ready,
    )
    try:
        authorized = await asyncio.shield(ready)
    except BaseException:
        await _cancel_live_transaction_lease_acquisition(task, release, ready)
        raise
    _live_resource_leases[lease_id] = (release, task)
    return lease_id, authorized


async def release_live_resource_lease(lease_id: str) -> bool:
    lease = _live_resource_leases.get(lease_id)
    if lease is None:
        return False
    release, task = lease
    release.set()
    await task
    return True


async def is_live_resource_lease_active(lease_id: str) -> bool:
    lease = _live_resource_leases.get(lease_id)
    return lease is not None and not lease[1].done()


async def release_all_live_resource_leases() -> None:
    leases = list(_live_resource_leases.values())
    for release, _ in leases:
        release.set()
    if leases:
        await asyncio.gather(*(task for _, task in leases), return_exceptions=True)


async def acquire_agent_graph_attachment_lease(graph_ids: list[str]) -> str:
    release = asyncio.Event()
    ready = asyncio.get_running_loop().create_future()
    lease_id = str(uuid4())

    async def hold() -> None:
        try:
            async with _live_transaction_lease_slot("attachment") as client:
                async with agent_graph_attachment_barriers(
                    sorted(set(graph_ids)), client=client
                ):
                    if not ready.done():
                        ready.set_result(None)
                    await release.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            else:
                logger.exception("Graph attachment lease %s failed", lease_id)
        finally:
            _graph_attachment_leases.pop(lease_id, None)

    task = _track_live_transaction_lease_task(
        asyncio.create_task(hold(), name=f"graph-attachment-lease:{lease_id}"),
        release,
        ready,
    )
    try:
        await asyncio.shield(ready)
    except BaseException:
        await _cancel_live_transaction_lease_acquisition(task, release, ready)
        raise
    _graph_attachment_leases[lease_id] = (release, task)
    return lease_id


async def release_agent_graph_attachment_lease(lease_id: str) -> bool:
    lease = _graph_attachment_leases.get(lease_id)
    if lease is None:
        return False
    release, task = lease
    release.set()
    await task
    return True


async def is_agent_graph_attachment_lease_active(lease_id: str) -> bool:
    lease = _graph_attachment_leases.get(lease_id)
    return lease is not None and not lease[1].done()


async def release_all_agent_graph_attachment_leases() -> None:
    leases = list(_graph_attachment_leases.values())
    for release, _ in leases:
        release.set()
    if leases:
        await asyncio.gather(*(task for _, task in leases), return_exceptions=True)


async def acquire_alert_condition_delivery_lease(condition_ids: list[str]) -> str:
    release = asyncio.Event()
    ready = asyncio.get_running_loop().create_future()
    lease_id = str(uuid4())

    async def hold() -> None:
        try:
            async with _live_transaction_lease_slot("delivery") as client:
                async with alert_condition_delivery_barriers(
                    condition_ids, client=client
                ):
                    if not ready.done():
                        ready.set_result(None)
                    await release.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            else:
                logger.exception("Alert condition lease %s failed", lease_id)
        finally:
            _alert_condition_leases.pop(lease_id, None)

    task = _track_live_transaction_lease_task(
        asyncio.create_task(hold(), name=f"alert-condition-lease:{lease_id}"),
        release,
        ready,
    )
    try:
        await asyncio.shield(ready)
    except BaseException:
        await _cancel_live_transaction_lease_acquisition(task, release, ready)
        raise
    _alert_condition_leases[lease_id] = (release, task)
    return lease_id


async def release_alert_condition_delivery_lease(lease_id: str) -> bool:
    lease = _alert_condition_leases.get(lease_id)
    if lease is None:
        return False
    release, task = lease
    release.set()
    await task
    return True


async def is_alert_condition_delivery_lease_active(lease_id: str) -> bool:
    lease = _alert_condition_leases.get(lease_id)
    return lease is not None and not lease[1].done()


async def acquire_store_listing_version_delivery_lease(version_id: str) -> str:
    release = asyncio.Event()
    ready = asyncio.get_running_loop().create_future()
    lease_id = str(uuid4())

    async def hold() -> None:
        try:
            async with _live_transaction_lease_slot("delivery") as client:
                async with client.tx(timeout=LIVE_ACTION_TRANSACTION_TIMEOUT) as tx:
                    rows = await tx.query_raw(
                        'SELECT "id" FROM "StoreListingVersion" '
                        'WHERE "id" = $1 FOR SHARE',
                        version_id,
                    )
                    if not rows:
                        raise ValueError("Store listing version no longer exists")
                    if not ready.done():
                        ready.set_result(None)
                    await release.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            else:
                logger.exception("Store listing version lease %s failed", lease_id)
        finally:
            _store_listing_version_leases.pop(lease_id, None)

    task = _track_live_transaction_lease_task(
        asyncio.create_task(hold(), name=f"store-listing-version-lease:{lease_id}"),
        release,
        ready,
    )
    try:
        await asyncio.shield(ready)
    except BaseException:
        await _cancel_live_transaction_lease_acquisition(task, release, ready)
        raise
    _store_listing_version_leases[lease_id] = (release, task)
    return lease_id


async def release_store_listing_version_delivery_lease(lease_id: str) -> bool:
    lease = _store_listing_version_leases.get(lease_id)
    if lease is None:
        return False
    release, task = lease
    release.set()
    await task
    return True


async def is_store_listing_version_delivery_lease_active(lease_id: str) -> bool:
    lease = _store_listing_version_leases.get(lease_id)
    return lease is not None and not lease[1].done()


async def release_all_notification_source_leases() -> None:
    leases = [
        *_alert_condition_leases.values(),
        *_store_listing_version_leases.values(),
    ]
    for release, _ in leases:
        release.set()
    if leases:
        await asyncio.gather(*(task for _, task in leases), return_exceptions=True)


@asynccontextmanager
async def live_org_permission_barrier(
    user_id: str,
    organization_id: str,
    action: OrgAction,
) -> AsyncIterator[bool]:
    async with _live_context_barrier(user_id, organization_id, None) as ctx:
        yield ctx is not None and check_org_permission(ctx, action)


@asynccontextmanager
async def live_org_context_barrier(
    user_id: str,
    organization_id: str,
) -> AsyncIterator[RequestContext | None]:
    async with _live_context_barrier(user_id, organization_id, None) as ctx:
        yield ctx


@asynccontextmanager
async def live_actor_org_context_barrier(
    user_id: str,
    organization_id: str,
) -> AsyncIterator[RequestContext | None]:
    async with _live_actor_context(user_id, organization_id) as client:
        yield await _get_live_context(user_id, organization_id, None, client)


@asynccontextmanager
async def live_actor_org_permission_barrier(
    user_id: str,
    organization_id: str,
    action: OrgAction,
) -> AsyncIterator[bool]:
    async with live_actor_org_context_barrier(user_id, organization_id) as ctx:
        yield ctx is not None and check_org_permission(ctx, action)


@asynccontextmanager
async def live_actor_org_or_team_permission_barrier(
    user_id: str,
    organization_id: str,
    team_id: str,
    org_action: OrgAction,
    team_action: TeamAction,
) -> AsyncIterator[bool]:
    async with _live_actor_context(user_id, organization_id) as client:
        org_ctx = await _get_live_context(user_id, organization_id, None, client)
        if org_ctx is None:
            yield False
            return
        if check_org_permission(org_ctx, org_action):
            yield True
            return
        team_ctx = await _get_live_context(user_id, organization_id, team_id, client)
        yield team_ctx is not None and check_team_permission(team_ctx, team_action)


@asynccontextmanager
async def live_resource_permission_barrier(
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
    org_action: OrgAction,
    team_action: TeamAction,
) -> AsyncIterator[bool]:
    if organization_id is None:
        yield True
        return
    async with _live_context_barrier(user_id, organization_id, team_id) as ctx:
        yield (
            ctx is not None
            and check_org_permission(ctx, org_action)
            and (team_id is None or check_team_permission(ctx, team_action))
        )


@asynccontextmanager
async def agent_graph_attachment_barrier(graph_id: str) -> AsyncIterator[None]:
    async with agent_graph_attachment_barriers([graph_id]):
        yield


@asynccontextmanager
async def alert_condition_delivery_barriers(
    condition_ids: list[str] | set[str] | tuple[str, ...],
    client: Prisma | None = None,
) -> AsyncIterator[None]:
    request_client = client or prisma
    async with request_client.tx(timeout=LIVE_ACTION_TRANSACTION_TIMEOUT) as tx:
        for condition_id in sorted(set(condition_ids)):
            await _lock_advisory_scope(
                tx, f"alert-condition:{condition_id}", shared=True
            )
        yield


@asynccontextmanager
async def alert_condition_mutation_barriers(
    condition_ids: list[str] | set[str] | tuple[str, ...],
) -> AsyncIterator[None]:
    async with prisma.tx(timeout=LIVE_ACTION_TRANSACTION_TIMEOUT) as tx:
        for condition_id in sorted(set(condition_ids)):
            await _lock_advisory_scope(tx, f"alert-condition:{condition_id}")
        yield


@asynccontextmanager
async def alert_condition_identity_mutation_barrier(
    user_id: str, cause_key: str
) -> AsyncIterator[None]:
    async with prisma.tx(timeout=LIVE_ACTION_TRANSACTION_TIMEOUT) as tx:
        await _lock_advisory_scope(
            tx, f"alert-condition-identity:{user_id}:{cause_key}"
        )
        yield


@asynccontextmanager
async def agent_graph_attachment_mutation_barrier(
    graph_id: str,
) -> AsyncIterator[None]:
    async with _request_or_direct_transaction() as tx:
        await _lock_advisory_scope(tx, f"agent-graph:{graph_id}")
        active_graphs = _active_graph_scopes.get()
        token = _active_graph_scopes.set(active_graphs | {graph_id})
        try:
            yield
        finally:
            _active_graph_scopes.reset(token)


@asynccontextmanager
async def agent_graph_attachment_barriers(
    graph_ids: list[str] | set[str] | tuple[str, ...],
    client: Prisma | None = None,
) -> AsyncIterator[None]:
    active_graphs = _active_graph_scopes.get()
    graph_ids_to_lock = sorted(set(graph_ids) - active_graphs)
    if not graph_ids_to_lock:
        yield
        return

    async with _request_or_direct_transaction(client) as tx:
        for graph_id in graph_ids_to_lock:
            await _lock_advisory_scope(tx, f"agent-graph:{graph_id}", shared=True)
        token = _active_graph_scopes.set(active_graphs | set(graph_ids_to_lock))
        try:
            yield
        finally:
            _active_graph_scopes.reset(token)


@asynccontextmanager
async def live_agent_graph_access_barrier(
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
    access: ResourceAccess,
    graph_id: str,
    graph_version: int,
) -> AsyncIterator[bool]:
    if organization_id is None:
        async with agent_graph_attachment_barrier(graph_id):
            yield True
        return

    scope = (user_id, organization_id, team_id)
    active_scopes = _active_live_scopes.get()
    active_graphs = _active_graph_scopes.get()
    async with _request_or_direct_transaction() as tx:
        if scope in active_scopes:
            ctx = await _get_live_context(user_id, organization_id, team_id, tx)
        else:
            await _lock_live_scope(tx, user_id, organization_id, team_id)
            ctx = await _get_live_context(user_id, organization_id, team_id, tx)
        if graph_id not in active_graphs:
            await _lock_advisory_scope(tx, f"agent-graph:{graph_id}", shared=True)
        graph = await tx.agentgraph.find_first(
            where={
                "id": graph_id,
                "version": graph_version,
                "organizationId": organization_id,
                "teamId": team_id,
            }
        )
        tenancy_token = None
        graph_token = None
        if scope not in active_scopes:
            tenancy_token = _active_live_scopes.set(active_scopes | {scope})
        if graph_id not in active_graphs:
            graph_token = _active_graph_scopes.set(active_graphs | {graph_id})
        try:
            yield graph is not None and _allows_resource_access(ctx, team_id, access)
        finally:
            if graph_token is not None:
                _active_graph_scopes.reset(graph_token)
            if tenancy_token is not None:
                _active_live_scopes.reset(tenancy_token)


def visibility_filter(
    user_id: str,
    organization_id: str | None,
    team_ids: list[str],
    *,
    user_field: str = "userId",
    org_field: str = "organizationId",
    team_field: str = "teamId",
    team_id_restriction: str | None = None,
) -> dict:
    """Build a Prisma OR-clause implementing the visibility rules above.

    With no org context (``organization_id`` is None) this degrades to
    plain personal ownership, preserving pre-org behaviour for internal
    callers that don't resolve a RequestContext.
    """
    if organization_id is None:
        return {user_field: user_id}
    if team_id_restriction is not None:
        return {
            org_field: organization_id,
            team_field: team_id_restriction,
        }

    return {
        "OR": [
            {
                user_field: user_id,
                org_field: None,
            },
            # Org-home rows: visible to every member of the org.
            {org_field: organization_id, team_field: None},
            # Team rows: visible to members of those teams.
            *(
                [{org_field: organization_id, team_field: {"in": team_ids}}]
                if team_ids
                else []
            ),
        ]
    }
