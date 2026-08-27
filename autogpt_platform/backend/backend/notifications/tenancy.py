import logging
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import TypeVar

from backend.data.db_accessors import LiveResourceAccessRevoked, LiveResourceLeaseGuard
from backend.data.notifications import NotificationScope
from backend.util.clients import get_database_manager_async_client

logger = logging.getLogger(__name__)
T = TypeVar("T")


def normalize_scopes(scopes: list[NotificationScope]) -> list[NotificationScope]:
    keys = {(scope.organization_id, scope.team_id) for scope in scopes}
    return [
        NotificationScope(organization_id=organization_id, team_id=team_id)
        for organization_id, team_id in sorted(
            keys, key=lambda scope: tuple(part or "" for part in scope)
        )
    ]


def scope_keys(scopes: list[NotificationScope]) -> set[tuple[str | None, str | None]]:
    return {(scope.organization_id, scope.team_id) for scope in scopes}


@asynccontextmanager
async def authorized_notification_scopes(
    user_id: str,
    candidate_scopes: list[NotificationScope],
) -> AsyncIterator[tuple[list[NotificationScope], tuple[LiveResourceLeaseGuard, ...]]]:
    db_client = get_database_manager_async_client(should_retry=False)
    normalized = normalize_scopes(candidate_scopes)
    authorized = [scope for scope in normalized if scope.organization_id is None]
    remote_scopes: list[tuple[str, str | None]] = [
        (scope.organization_id, scope.team_id)
        for scope in normalized
        if scope.organization_id is not None
    ]
    lease_id: str | None = None
    guards: tuple[LiveResourceLeaseGuard, ...] = ()

    try:
        if remote_scopes:
            (
                lease_id,
                remote_authorized,
            ) = await db_client.acquire_live_resource_scopes_lease(
                user_id,
                remote_scopes,
                "execute",
            )
            authorized.extend(
                NotificationScope(organization_id=organization_id, team_id=team_id)
                for organization_id, team_id in remote_authorized
            )
            guards = (LiveResourceLeaseGuard(db_client, lease_id),)
            authorized_remote_keys: set[tuple[str, str | None]] = {
                (scope.organization_id, scope.team_id)
                for scope in authorized
                if scope.organization_id is not None
            }
            denied = set(remote_scopes) - authorized_remote_keys
            if denied:
                logger.info(
                    "Dropping %s revoked notification scopes for user %s",
                    len(denied),
                    user_id,
                )

        yield normalize_scopes(authorized), guards
    finally:
        if lease_id is not None and not await db_client.release_live_resource_lease(
            lease_id
        ):
            raise LiveResourceAccessRevoked("notification_lease_lost")


async def run_with_notification_guards(
    action: Awaitable[T], guards: tuple[LiveResourceLeaseGuard, ...]
) -> T:
    guarded_action = action
    for guard in reversed(guards):
        guarded_action = guard.run(guarded_action)
    return await guarded_action
