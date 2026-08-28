import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from contextvars import Context, ContextVar
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict

from backend.data import db
from backend.data.tenancy import ResourceAccess, context_without_live_tenancy_scopes


class LiveResourceAccessRevoked(RuntimeError):
    pass


T = TypeVar("T")


class LiveResourceLeaseGuard:
    def __init__(self, lease_db, lease_id: str) -> None:
        self._lease_db = lease_db
        self.lease_id = lease_id

    def __bool__(self) -> Literal[True]:
        return True

    async def validate(self) -> None:
        try:
            active = await self._lease_db.is_live_resource_lease_active(self.lease_id)
        except Exception as exc:
            raise LiveResourceAccessRevoked("workspace_lease_lost") from exc
        if not active:
            raise LiveResourceAccessRevoked("workspace_lease_lost")

    async def _wait_for_loss(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            await self.validate()

    async def run(self, action: Awaitable[T], *, context: Context | None = None) -> T:
        start = asyncio.Event()
        action_started = False

        async def run_after_initial_validation() -> T:
            nonlocal action_started
            await start.wait()
            action_started = True
            return await action

        action_task = asyncio.create_task(
            run_after_initial_validation(),
            context=context,
        )
        loss_task = asyncio.create_task(self._wait_for_loss())
        try:
            await self.validate()
            start.set()
            done, _ = await asyncio.wait(
                (action_task, loss_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if loss_task in done:
                action_task.cancel()
                await asyncio.gather(action_task, return_exceptions=True)
                await loss_task
            result = await action_task
            await self.validate()
            return result
        finally:
            if not action_task.done():
                action_task.cancel()
            if not loss_task.done():
                loss_task.cancel()
            await asyncio.gather(action_task, loss_task, return_exceptions=True)
            if not action_started and inspect.iscoroutine(action):
                action.close()


class _ActiveLiveResourceLease(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    user_id: str
    organization_id: str | None
    team_id: str | None
    access: ResourceAccess
    guard: LiveResourceLeaseGuard


_active_live_resource_leases: ContextVar[tuple[_ActiveLiveResourceLease, ...]] = (
    ContextVar("active_live_resource_leases", default=())
)


def context_without_live_leases() -> Context:
    context = context_without_live_tenancy_scopes()
    context.run(_active_live_resource_leases.set, ())
    return context


async def run_with_live_resource_lease_guard(
    guard: LiveResourceLeaseGuard,
    *,
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
    access: ResourceAccess,
    action: Awaitable[T],
) -> T:
    active = _ActiveLiveResourceLease(
        user_id=user_id,
        organization_id=organization_id,
        team_id=team_id,
        access=access,
        guard=guard,
    )
    token = _active_live_resource_leases.set(
        (*_active_live_resource_leases.get(), active)
    )
    try:
        return await guard.run(action)
    finally:
        _active_live_resource_leases.reset(token)


class AgentGraphAttachmentLeaseGuard:
    def __init__(self, lease_db, lease_id: str) -> None:
        self._lease_db = lease_db
        self.lease_id = lease_id

    async def validate(self) -> None:
        try:
            active = await self._lease_db.is_agent_graph_attachment_lease_active(
                self.lease_id
            )
        except Exception as exc:
            raise LiveResourceAccessRevoked("graph_attachment_lease_lost") from exc
        if not active:
            raise LiveResourceAccessRevoked("graph_attachment_lease_lost")

    async def _wait_for_loss(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            await self.validate()

    async def run(self, action: Awaitable[T]) -> T:
        start = asyncio.Event()
        action_started = False

        async def run_after_initial_validation() -> T:
            nonlocal action_started
            await start.wait()
            action_started = True
            return await action

        action_task = asyncio.create_task(run_after_initial_validation())
        loss_task = asyncio.create_task(self._wait_for_loss())
        try:
            await self.validate()
            start.set()
            done, _ = await asyncio.wait(
                (action_task, loss_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if loss_task in done:
                action_task.cancel()
                await asyncio.gather(action_task, return_exceptions=True)
                await loss_task
            result = await action_task
            await self.validate()
            return result
        finally:
            if not action_task.done():
                action_task.cancel()
            if not loss_task.done():
                loss_task.cancel()
            await asyncio.gather(action_task, loss_task, return_exceptions=True)
            if not action_started and inspect.iscoroutine(action):
                action.close()


class AlertConditionDeliveryLeaseGuard(AgentGraphAttachmentLeaseGuard):
    async def validate(self) -> None:
        try:
            active = await self._lease_db.is_alert_condition_delivery_lease_active(
                self.lease_id
            )
        except Exception as exc:
            raise LiveResourceAccessRevoked("alert_condition_lease_lost") from exc
        if not active:
            raise LiveResourceAccessRevoked("alert_condition_lease_lost")


class StoreListingVersionDeliveryLeaseGuard(AgentGraphAttachmentLeaseGuard):
    async def validate(self) -> None:
        try:
            active = (
                await self._lease_db.is_store_listing_version_delivery_lease_active(
                    self.lease_id
                )
            )
        except Exception as exc:
            raise LiveResourceAccessRevoked("store_listing_version_lease_lost") from exc
        if not active:
            raise LiveResourceAccessRevoked("store_listing_version_lease_lost")


async def require_exact_chat_session_scope(
    session_id: str,
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
) -> None:
    session = await chat_db().get_chat_session_metadata(session_id)
    if session is None or (
        session.user_id,
        session.organization_id,
        session.team_id,
    ) != (user_id, organization_id, team_id):
        raise LiveResourceAccessRevoked("workspace_access_revoked")


@asynccontextmanager
async def live_resource_lease(
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
    access: ResourceAccess,
) -> AsyncIterator[LiveResourceLeaseGuard | Literal[False]]:
    for active in reversed(_active_live_resource_leases.get()):
        if (
            active.user_id,
            active.organization_id,
            active.team_id,
            active.access,
        ) == (user_id, organization_id, team_id, access):
            await active.guard.validate()
            yield active.guard
            return

    lease_db = credit_db()
    lease_id = await lease_db.acquire_live_resource_lease(
        user_id, organization_id, team_id, access
    )
    if lease_id is None:
        yield False
        return
    guard = LiveResourceLeaseGuard(lease_db, lease_id)
    active = _ActiveLiveResourceLease(
        user_id=user_id,
        organization_id=organization_id,
        team_id=team_id,
        access=access,
        guard=guard,
    )
    token = _active_live_resource_leases.set(
        (*_active_live_resource_leases.get(), active)
    )
    try:
        yield guard
    finally:
        try:
            released = await lease_db.release_live_resource_lease(lease_id)
            if not released:
                raise LiveResourceAccessRevoked("workspace_lease_lost")
        finally:
            _active_live_resource_leases.reset(token)


@asynccontextmanager
async def live_resource_scopes_lease(
    user_id: str,
    scopes: list[tuple[str, str | None]],
    access: ResourceAccess,
) -> AsyncIterator[list[tuple[str, str | None]]]:
    normalized = sorted(set(scopes), key=lambda scope: (scope[0], scope[1] or ""))
    if not normalized:
        yield []
        return

    lease_db = credit_db()
    lease_id, authorized = await lease_db.acquire_live_resource_scopes_lease(
        user_id, normalized, access
    )
    guard = LiveResourceLeaseGuard(lease_db, lease_id)
    active = tuple(
        _ActiveLiveResourceLease(
            user_id=user_id,
            organization_id=organization_id,
            team_id=team_id,
            access=access,
            guard=guard,
        )
        for organization_id, team_id in authorized
    )
    token = _active_live_resource_leases.set(
        (*_active_live_resource_leases.get(), *active)
    )
    try:
        yield authorized
    finally:
        try:
            released = await lease_db.release_live_resource_lease(lease_id)
            if not released:
                raise LiveResourceAccessRevoked("workspace_lease_lost")
        finally:
            _active_live_resource_leases.reset(token)


@asynccontextmanager
async def agent_graph_attachment_lease(
    graph_ids: list[str],
) -> AsyncIterator[AgentGraphAttachmentLeaseGuard]:
    lease_db = credit_db()
    lease_id = await lease_db.acquire_agent_graph_attachment_lease(graph_ids)
    guard = AgentGraphAttachmentLeaseGuard(lease_db, lease_id)
    try:
        yield guard
    finally:
        released = await lease_db.release_agent_graph_attachment_lease(lease_id)
        if not released:
            raise LiveResourceAccessRevoked("graph_attachment_lease_lost")


@asynccontextmanager
async def alert_condition_delivery_lease(
    condition_ids: list[str],
) -> AsyncIterator[AlertConditionDeliveryLeaseGuard]:
    lease_db = credit_db()
    lease_id = await lease_db.acquire_alert_condition_delivery_lease(condition_ids)
    guard = AlertConditionDeliveryLeaseGuard(lease_db, lease_id)
    try:
        yield guard
    finally:
        released = await lease_db.release_alert_condition_delivery_lease(lease_id)
        if not released:
            raise LiveResourceAccessRevoked("alert_condition_lease_lost")


@asynccontextmanager
async def store_listing_version_delivery_lease(
    version_id: str,
) -> AsyncIterator[StoreListingVersionDeliveryLeaseGuard]:
    lease_db = credit_db()
    lease_id = await lease_db.acquire_store_listing_version_delivery_lease(version_id)
    guard = StoreListingVersionDeliveryLeaseGuard(lease_db, lease_id)
    try:
        yield guard
    finally:
        released = await lease_db.release_store_listing_version_delivery_lease(lease_id)
        if not released:
            raise LiveResourceAccessRevoked("store_listing_version_lease_lost")


def chat_db():
    if db.is_connected():
        from backend.copilot import db as _chat_db

        chat_db = _chat_db
    else:
        from backend.util.clients import get_database_manager_async_client

        chat_db = get_database_manager_async_client()

    return chat_db


def experts_db():
    if db.is_connected():
        from backend.api.features.experts import experts_db as _experts_db

        experts_db = _experts_db
    else:
        from backend.util.clients import get_database_manager_async_client

        experts_db = get_database_manager_async_client()

    return experts_db


def graph_db():
    if db.is_connected():
        from backend.data import graph as _graph_db

        graph_db = _graph_db
    else:
        from backend.util.clients import get_database_manager_async_client

        graph_db = get_database_manager_async_client()

    return graph_db


def library_db():
    if db.is_connected():
        from backend.api.features.library import db as _library_db

        library_db = _library_db
    else:
        from backend.util.clients import get_database_manager_async_client

        library_db = get_database_manager_async_client()

    return library_db


def store_db():
    if db.is_connected():
        from backend.api.features.store import db as _store_db

        store_db = _store_db
    else:
        from backend.util.clients import get_database_manager_async_client

        store_db = get_database_manager_async_client()

    return store_db


def triggers_db():
    if db.is_connected():
        from backend.api.features.library import triggers as _triggers_db

        triggers_db = _triggers_db
    else:
        from backend.util.clients import get_database_manager_async_client

        triggers_db = get_database_manager_async_client()

    return triggers_db


def search():
    if db.is_connected():
        from backend.api.features.search import hybrid_search as _search

        search = _search
    else:
        from backend.util.clients import get_database_manager_async_client

        search = get_database_manager_async_client()

    return search


def execution_db():
    if db.is_connected():
        from backend.data import execution as _execution_db

        execution_db = _execution_db
    else:
        from backend.util.clients import get_database_manager_async_client

        execution_db = get_database_manager_async_client()

    return execution_db


def user_db():
    if db.is_connected():
        from backend.data import user as _user_db

        user_db = _user_db
    else:
        from backend.util.clients import get_database_manager_async_client

        user_db = get_database_manager_async_client()

    return user_db


def understanding_db():
    if db.is_connected():
        from backend.data import understanding as _understanding_db

        understanding_db = _understanding_db
    else:
        from backend.util.clients import get_database_manager_async_client

        understanding_db = get_database_manager_async_client()

    return understanding_db


def workspace_db():
    if db.is_connected():
        from backend.data import workspace as _workspace_db

        workspace_db = _workspace_db
    else:
        from backend.util.clients import get_database_manager_async_client

        workspace_db = get_database_manager_async_client()

    return workspace_db


def review_db():
    if db.is_connected():
        from backend.data import human_review as _review_db

        review_db = _review_db
    else:
        from backend.util.clients import get_database_manager_async_client

        review_db = get_database_manager_async_client()

    return review_db


def credit_db():
    if db.is_connected():
        from backend.data import db_manager as _credit_db

        credit_db = _credit_db
    else:
        from backend.util.clients import get_database_manager_async_client

        credit_db = get_database_manager_async_client()

    return credit_db


def platform_cost_db():
    if db.is_connected():
        from backend.data import platform_cost as _platform_cost_db

        platform_cost_db = _platform_cost_db
    else:
        from backend.util.clients import get_database_manager_async_client

        platform_cost_db = get_database_manager_async_client()

    return platform_cost_db


def orgs_db():
    if db.is_connected():
        from backend.api.features.orgs import db as _orgs_db

        orgs_db = _orgs_db
    else:
        from backend.util.clients import get_database_manager_async_client

        orgs_db = get_database_manager_async_client()

    return orgs_db


def platform_linking_db():
    if db.is_connected():
        from backend.platform_linking import db as _platform_linking_db

        platform_linking_db = _platform_linking_db
    else:
        from backend.util.clients import get_database_manager_async_client

        platform_linking_db = get_database_manager_async_client()

    return platform_linking_db


def bot_analytics_db():
    if db.is_connected():
        from backend.data import bot_analytics as _bot_analytics_db

        bot_analytics_db = _bot_analytics_db
    else:
        from backend.util.clients import get_database_manager_async_client

        bot_analytics_db = get_database_manager_async_client()

    return bot_analytics_db


def bot_installs_db():
    if db.is_connected():
        from backend.data import bot_installs as _bot_installs_db

        bot_installs_db = _bot_installs_db
    else:
        from backend.util.clients import get_database_manager_async_client

        bot_installs_db = get_database_manager_async_client()

    return bot_installs_db
