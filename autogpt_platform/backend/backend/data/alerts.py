"""Persistence for live alert conditions.

One row per condition that needs the user's attention. `cause_key` identifies
the condition, so re-raising the same problem updates the existing row instead
of adding a duplicate, and resolving it can find the row by key alone.

The lifecycle the alert engine drives off this table:

    PENDING  ── debounce window elapses, under the daily cap ──▶ SENT
       │                                                          │
       │  condition clears first                                  │  re-raised
       ▼                                                          ▼  within 24h
    RESOLVED                                                   DEFERRED
                                                                  │
                                                     folded into the next
                                                     Briefing's attention block
"""

import logging
from contextlib import AsyncExitStack
from datetime import datetime, timedelta, timezone

from prisma.enums import AlertCause, AlertConditionStatus
from prisma.models import AgentGraphExecution, AlertCondition
from prisma.types import AlertConditionWhereInput
from pydantic import BaseModel

from backend.data.db import prisma
from backend.data.notifications import NotificationScope
from backend.data.tenancy import (
    agent_graph_attachment_barriers,
    alert_condition_identity_mutation_barrier,
    alert_condition_mutation_barriers,
)
from backend.util.exceptions import DatabaseError
from backend.util.json import SafeJson
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[Alerts]")

# The same cause never re-alerts inside this window; if it persists it
# escalates into the next Briefing's attention block.
ALERT_DEDUPE_WINDOW = timedelta(hours=24)

# Only ever used to bound a counting read; well above the real daily cap so the
# count is never silently truncated below it.
MAX_ALERT_EMAILS_PER_DAY_CEILING = 100

# Statuses that still describe a live problem, in the sense that the Briefing
# should tell the user about it.
LIVE_STATUSES = [
    AlertConditionStatus.PENDING,
    AlertConditionStatus.SENT,
    AlertConditionStatus.DEFERRED,
]


class AlertConditionDTO(BaseModel):
    id: str
    user_id: str
    cause: AlertCause
    cause_key: str
    data: dict
    status: AlertConditionStatus
    created_at: datetime
    sent_at: datetime | None
    briefed_at: datetime | None
    organization_id: str | None = None
    team_id: str | None = None
    source_graph_execution_id: str | None = None

    @staticmethod
    def from_db(model: AlertCondition) -> "AlertConditionDTO":
        return AlertConditionDTO(
            id=model.id,
            user_id=model.userId,
            cause=model.cause,
            cause_key=model.causeKey,
            data=dict(model.data),
            status=model.status,
            created_at=model.createdAt,
            sent_at=model.sentAt,
            briefed_at=model.briefedAt,
            organization_id=model.organizationId,
            team_id=model.teamId,
            source_graph_execution_id=model.sourceGraphExecutionId,
        )


async def raise_alert_condition(
    user_id: str,
    cause: AlertCause,
    cause_key: str,
    data: dict,
    organization_id: str | None = None,
    team_id: str | None = None,
    source_graph_execution_id: str | None = None,
) -> AlertConditionDTO:
    """Record a live condition.

    A condition alerted on within the last 24 hours does not re-alert: it
    becomes DEFERRED and escalates into the next Briefing's attention block
    instead of re-pinging the inbox.
    """
    if team_id is not None and organization_id is None:
        raise DatabaseError("Alert team scope requires an organization")
    if organization_id is not None and source_graph_execution_id is None:
        raise DatabaseError("Organization alert requires a source execution")

    dedupe_since = datetime.now(tz=timezone.utc) - ALERT_DEDUPE_WINDOW
    try:
        payload = SafeJson(data)
        async with AsyncExitStack() as stack:
            source = None
            if source_graph_execution_id is not None:
                source = await AgentGraphExecution.prisma().find_unique(
                    where={"id": source_graph_execution_id}
                )
                if source is None:
                    raise DatabaseError("Alert source execution is no longer live")
                await stack.enter_async_context(
                    agent_graph_attachment_barriers([source.agentGraphId])
                )
                source = await AgentGraphExecution.prisma().find_unique(
                    where={"id": source_graph_execution_id}
                )
                if (
                    source is None
                    or source.isDeleted
                    or (
                        source.id,
                        source.userId,
                        source.organizationId,
                        source.teamId,
                    )
                    != (
                        source_graph_execution_id,
                        user_id,
                        organization_id,
                        team_id,
                    )
                ):
                    raise DatabaseError("Alert source execution is no longer live")

            await stack.enter_async_context(
                alert_condition_identity_mutation_barrier(user_id, cause_key)
            )
            existing = await AlertCondition.prisma().find_unique(
                where={"userId_causeKey": {"userId": user_id, "causeKey": cause_key}},
                include={"SourceGraphExecution": True},
            )
            if existing is None:
                created = await AlertCondition.prisma().create(
                    data={
                        "userId": user_id,
                        "cause": cause,
                        "causeKey": cause_key,
                        "data": payload,
                        "organizationId": organization_id,
                        "teamId": team_id,
                        "sourceGraphExecutionId": source_graph_execution_id,
                    }
                )
                return AlertConditionDTO.from_db(created)

            await stack.enter_async_context(
                alert_condition_mutation_barriers([existing.id])
            )
            existing = await AlertCondition.prisma().find_unique(
                where={"id": existing.id},
                include={"SourceGraphExecution": True},
            )
            if existing is None:
                raise DatabaseError("Alert condition vanished before update")
            if (
                existing.organizationId,
                existing.teamId,
                existing.sourceGraphExecutionId,
            ) != (organization_id, team_id, source_graph_execution_id):
                raise DatabaseError(
                    f"Alert condition {existing.id} cannot change authorization scope"
                )
            execution = existing.SourceGraphExecution
            if source_graph_execution_id is not None and (
                execution is None
                or execution.isDeleted
                or (
                    execution.id,
                    execution.userId,
                    execution.organizationId,
                    execution.teamId,
                )
                != (
                    source_graph_execution_id,
                    user_id,
                    organization_id,
                    team_id,
                )
            ):
                raise DatabaseError("Alert source execution is no longer live")

            recently_alerted = (
                existing.sentAt is not None and existing.sentAt > dedupe_since
            )
            status = (
                AlertConditionStatus.DEFERRED
                if recently_alerted
                else AlertConditionStatus.PENDING
            )
            if existing.status is AlertConditionStatus.DEFERRED:
                status = AlertConditionStatus.DEFERRED

            updated = await AlertCondition.prisma().update(
                where={"id": existing.id},
                data={
                    "cause": cause,
                    "data": payload,
                    "status": status,
                    "resolvedAt": None,
                    "briefedAt": None,
                },
            )
            if updated is None:
                raise DatabaseError(
                    f"Alert condition {existing.id} vanished mid-update"
                )
            return AlertConditionDTO.from_db(updated)
    except DatabaseError:
        raise
    except Exception as e:
        raise DatabaseError(
            f"Failed to raise alert condition {cause_key} for user {user_id}: {e}"
        ) from e


async def resolve_alert_condition(user_id: str, cause_key: str) -> bool:
    """Mark a condition fixed. Returns whether a live row was cleared.

    Called when the underlying problem goes away — including during the
    debounce window, which cancels the send outright, because an alert about a
    solved problem trains people to ignore alerts.
    """
    try:
        condition = await AlertCondition.prisma().find_unique(
            where={"userId_causeKey": {"userId": user_id, "causeKey": cause_key}}
        )
        if condition is None:
            return False
        async with alert_condition_mutation_barriers([condition.id]):
            cleared = await AlertCondition.prisma().update_many(
                where={
                    "id": condition.id,
                    "userId": user_id,
                    "causeKey": cause_key,
                    "status": {"in": LIVE_STATUSES},
                },
                data={
                    "status": AlertConditionStatus.RESOLVED,
                    "resolvedAt": datetime.now(tz=timezone.utc),
                },
            )
            return cleared > 0
    except Exception as e:
        raise DatabaseError(
            f"Failed to resolve alert condition {cause_key} for user {user_id}: {e}"
        ) from e


class MaturedAlertPage(BaseModel):
    """One keyset page of users with matured alerts.

    `exhausted` is set from the row count rather than the user count,
    because de-duplication shrinks the page.
    """

    user_ids: list[str]
    exhausted: bool


async def get_users_with_matured_alerts(
    matured_before: datetime, after_user_id: str | None = None, limit: int = 1000
) -> MaturedAlertPage:
    """Users holding at least one PENDING condition older than the debounce
    window, i.e. whose pending alerts are ready to go out as one email.

    Bounded and keyset-paged on `userId`. This runs every minute against every
    PENDING row on the platform; an unbounded `find_many` would pull each row's
    full JSONB payload across the wire once a minute forever.

    `distinct` is applied by the query engine after the take, so the page is
    requested by ordered `userId` and de-duplicated here. That makes the
    returned user count smaller than the rows read, so `exhausted` reports
    whether the *raw* page was short — the caller cannot infer it from the
    number of users, which is the normal case for anyone holding more than
    one condition.
    """
    try:
        where: AlertConditionWhereInput = {
            "status": AlertConditionStatus.PENDING,
            "createdAt": {"lt": matured_before},
        }
        if after_user_id:
            where["userId"] = {"gt": after_user_id}
        rows = await AlertCondition.prisma().find_many(
            where=where,
            order={"userId": "asc"},
            take=limit,
        )
        seen: list[str] = []
        for row in rows:
            if not seen or seen[-1] != row.userId:
                seen.append(row.userId)
        return MaturedAlertPage(user_ids=seen, exhausted=len(rows) < limit)
    except Exception as e:
        raise DatabaseError(f"Failed to list users with matured alerts: {e}") from e


# One user cannot have more distinct live causes than this before the email
# stops being readable; the rest stay PENDING for the next flush.
MAX_CONDITIONS_PER_EMAIL = 50


async def get_pending_alert_conditions(
    user_id: str,
    authorization_scopes: list[NotificationScope] | None = None,
) -> list[AlertConditionDTO]:
    try:
        where: AlertConditionWhereInput = {
            "userId": user_id,
            "status": AlertConditionStatus.PENDING,
        }
        _apply_scope_filter(where, authorization_scopes)
        rows = await AlertCondition.prisma().find_many(
            where=where,
            order={"createdAt": "asc"},
            take=MAX_CONDITIONS_PER_EMAIL,
        )
        return [AlertConditionDTO.from_db(row) for row in rows]
    except Exception as e:
        raise DatabaseError(
            f"Failed to get pending alert conditions for user {user_id}: {e}"
        ) from e


async def count_alerts_sent_since(user_id: str, since: datetime) -> int:
    """How many Alert *emails* this user has already had since `since`.

    Every condition coalesced into one email is stamped with the same `sentAt`
    (one `update_many` with one timestamp), so counting distinct timestamps
    counts emails, not conditions — which is what the daily cap is about.
    """
    try:
        rows = await AlertCondition.prisma().find_many(
            where={"userId": user_id, "sentAt": {"gte": since}},
            distinct=["sentAt"],
            take=MAX_ALERT_EMAILS_PER_DAY_CEILING,
        )
        return len(rows)
    except Exception as e:
        raise DatabaseError(
            f"Failed to count alerts sent for user {user_id}: {e}"
        ) from e


async def finalize_alert_delivery(
    user_id: str,
    condition_ids: list[str],
    authorization_scopes: list[NotificationScope],
    delivered_at: datetime,
) -> None:
    if not condition_ids:
        return
    try:
        await AlertCondition.prisma().update_many(
            where={
                "id": {"in": condition_ids},
                "userId": user_id,
                "status": AlertConditionStatus.PENDING,
                "OR": [_scope_where(scope) for scope in authorization_scopes],
            },
            data={
                "status": AlertConditionStatus.SENT,
                "sentAt": delivered_at,
            },
        )
    except Exception as e:
        raise DatabaseError(f"Failed to finalize alert delivery: {e}") from e


async def mark_alert_conditions_deferred(condition_ids: list[str]) -> None:
    """Overflow past the daily cap folds into the Briefing rather than being
    dropped — nothing actionable is ever silently lost."""
    if not condition_ids:
        return
    try:
        async with alert_condition_mutation_barriers(condition_ids):
            await AlertCondition.prisma().update_many(
                where={"id": {"in": condition_ids}},
                data={"status": AlertConditionStatus.DEFERRED},
            )
    except Exception as e:
        raise DatabaseError(f"Failed to defer alert conditions: {e}") from e


async def get_briefing_alert_conditions(
    user_id: str,
    authorization_scopes: list[NotificationScope] | None = None,
) -> list[AlertConditionDTO]:
    """Everything the next Briefing's attention block must absorb: conditions
    capped or deduped during the period, plus any still unresolved, minus the
    ones a previous Briefing already reported."""
    try:
        where: AlertConditionWhereInput = {
            "userId": user_id,
            "status": {"in": LIVE_STATUSES},
            "briefedAt": None,
        }
        _apply_scope_filter(where, authorization_scopes)
        rows = await AlertCondition.prisma().find_many(
            where=where,
            order={"createdAt": "asc"},
            take=MAX_CONDITIONS_PER_EMAIL,
        )
        return [AlertConditionDTO.from_db(row) for row in rows]
    except Exception as e:
        raise DatabaseError(
            f"Failed to get briefing alert conditions for user {user_id}: {e}"
        ) from e


async def finalize_briefing_delivery(
    user_id: str,
    condition_ids: list[str],
    authorization_scopes: list[NotificationScope],
    delivered_at: datetime,
    scheduled_for: datetime,
) -> None:
    try:
        async with prisma.tx() as tx:
            if condition_ids:
                await tx.alertcondition.update_many(
                    where={
                        "id": {"in": condition_ids},
                        "userId": user_id,
                        "briefedAt": None,
                        "OR": [_scope_where(scope) for scope in authorization_scopes],
                    },
                    data={"briefedAt": delivered_at},
                )
            await tx.user.update_many(
                where={
                    "id": user_id,
                    "OR": [
                        {"lastBriefingAt": None},
                        {"lastBriefingAt": {"lt": scheduled_for}},
                    ],
                },
                data={"lastBriefingAt": scheduled_for},
            )
    except Exception as e:
        raise DatabaseError(f"Failed to finalize briefing delivery: {e}") from e


async def get_alert_condition_source_graph_ids(
    user_id: str, condition_ids: list[str]
) -> list[str]:
    if not condition_ids:
        return []
    try:
        rows = await AlertCondition.prisma().find_many(
            where={"id": {"in": condition_ids}, "userId": user_id},
            include={"SourceGraphExecution": True},
        )
        return sorted(
            {
                row.SourceGraphExecution.agentGraphId
                for row in rows
                if row.SourceGraphExecution is not None
            }
        )
    except Exception as e:
        raise DatabaseError(f"Failed to load alert source graphs: {e}") from e


async def alert_condition_sources_are_live(
    user_id: str,
    condition_ids: list[str],
    authorization_scopes: list[NotificationScope],
) -> bool:
    return not await get_stale_alert_condition_ids(
        user_id,
        condition_ids,
        authorization_scopes,
        for_briefing=False,
    )


async def get_stale_alert_condition_ids(
    user_id: str,
    condition_ids: list[str],
    authorization_scopes: list[NotificationScope],
    for_briefing: bool,
) -> list[str]:
    if not condition_ids:
        return []
    try:
        where: AlertConditionWhereInput = {
            "id": {"in": condition_ids},
            "userId": user_id,
            "status": (
                {"in": LIVE_STATUSES} if for_briefing else AlertConditionStatus.PENDING
            ),
        }
        if for_briefing:
            where["briefedAt"] = None
        rows = await AlertCondition.prisma().find_many(
            where=where,
            include={"SourceGraphExecution": True},
        )
        stale = set(condition_ids) - {row.id for row in rows}
        allowed_scopes = {
            (scope.organization_id, scope.team_id) for scope in authorization_scopes
        }
        for row in rows:
            scope = (row.organizationId, row.teamId)
            if scope not in allowed_scopes:
                stale.add(row.id)
                continue
            execution = row.SourceGraphExecution
            if execution is None:
                if (
                    row.organizationId is not None
                    or row.sourceGraphExecutionId is not None
                ):
                    stale.add(row.id)
                continue
            if execution.isDeleted or (
                execution.id,
                execution.userId,
                execution.organizationId,
                execution.teamId,
            ) != (
                row.sourceGraphExecutionId,
                row.userId,
                row.organizationId,
                row.teamId,
            ):
                stale.add(row.id)
        return sorted(stale)
    except Exception as e:
        raise DatabaseError(f"Failed to validate alert sources: {e}") from e


async def get_pending_alert_condition_scopes(
    user_id: str,
) -> list[NotificationScope]:
    return await _get_alert_condition_scopes(
        user_id,
        {"status": AlertConditionStatus.PENDING},
    )


async def get_briefing_alert_condition_scopes(
    user_id: str,
) -> list[NotificationScope]:
    return await _get_alert_condition_scopes(
        user_id,
        {"status": {"in": LIVE_STATUSES}, "briefedAt": None},
    )


async def resolve_alert_conditions_for_scopes(
    user_id: str, scopes: list[NotificationScope]
) -> None:
    if not scopes:
        return
    try:
        rows = await AlertCondition.prisma().find_many(
            where={
                "userId": user_id,
                "status": {"in": LIVE_STATUSES},
                "OR": [_scope_where(scope) for scope in scopes],
            }
        )
        await resolve_alert_conditions_by_ids(user_id, [row.id for row in rows])
    except Exception as e:
        raise DatabaseError(
            f"Failed to resolve revoked alert scopes for user {user_id}: {e}"
        ) from e


async def resolve_alert_conditions_by_ids(
    user_id: str, condition_ids: list[str]
) -> None:
    if not condition_ids:
        return
    try:
        async with alert_condition_mutation_barriers(condition_ids):
            await AlertCondition.prisma().update_many(
                where={
                    "id": {"in": condition_ids},
                    "userId": user_id,
                    "status": {"in": LIVE_STATUSES},
                },
                data={
                    "status": AlertConditionStatus.RESOLVED,
                    "resolvedAt": datetime.now(tz=timezone.utc),
                },
            )
    except Exception as e:
        raise DatabaseError(f"Failed to resolve alert conditions: {e}") from e


async def resolve_alert_conditions_for_source_execution(
    source_graph_execution_id: str,
) -> None:
    try:
        rows = await AlertCondition.prisma().find_many(
            where={
                "sourceGraphExecutionId": source_graph_execution_id,
                "status": {"in": LIVE_STATUSES},
            }
        )
        condition_ids = [row.id for row in rows]
        if not condition_ids:
            return
        async with alert_condition_mutation_barriers(condition_ids):
            await AlertCondition.prisma().update_many(
                where={
                    "id": {"in": condition_ids},
                    "sourceGraphExecutionId": source_graph_execution_id,
                    "status": {"in": LIVE_STATUSES},
                },
                data={
                    "status": AlertConditionStatus.RESOLVED,
                    "resolvedAt": datetime.now(tz=timezone.utc),
                },
            )
    except Exception as e:
        raise DatabaseError(
            "Failed to resolve alert conditions for deleted execution"
        ) from e


async def _get_alert_condition_scopes(
    user_id: str, where_extra: AlertConditionWhereInput
) -> list[NotificationScope]:
    try:
        rows = await AlertCondition.prisma().find_many(
            where={"userId": user_id, **where_extra},
            distinct=["organizationId", "teamId"],
            order={"createdAt": "asc"},
        )
        return [
            NotificationScope(
                organization_id=row.organizationId,
                team_id=row.teamId,
            )
            for row in rows
        ]
    except Exception as e:
        raise DatabaseError(
            f"Failed to list alert scopes for user {user_id}: {e}"
        ) from e


def _apply_scope_filter(
    where: AlertConditionWhereInput,
    authorization_scopes: list[NotificationScope] | None,
) -> None:
    if authorization_scopes is not None:
        where["OR"] = [_scope_where(scope) for scope in authorization_scopes]


def _scope_where(scope: NotificationScope) -> AlertConditionWhereInput:
    return {
        "organizationId": scope.organization_id,
        "teamId": scope.team_id,
    }
