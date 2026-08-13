"""Persistence for live alert conditions.

One row per "the platform is blocked on the human" condition, keyed by a stable
`cause_key` so re-raising the same problem updates one row instead of piling up
duplicates, and resolving it can find the row without the payload.

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
from datetime import datetime, timedelta, timezone

from prisma.enums import AlertCause, AlertConditionStatus
from prisma.models import AlertCondition
from pydantic import BaseModel

from backend.util.exceptions import DatabaseError
from backend.util.json import SafeJson
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[Alerts]")

# The same cause never re-alerts inside this window; if it persists it
# escalates into the next Briefing's attention block.
ALERT_DEDUPE_WINDOW = timedelta(hours=24)

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
        )


async def raise_condition(
    user_id: str,
    cause: AlertCause,
    cause_key: str,
    data: dict,
) -> AlertConditionDTO:
    """Record a live condition.

    A condition alerted on within the last 24 hours does not re-alert: it
    becomes DEFERRED and escalates into the next Briefing's attention block
    instead of re-pinging the inbox.
    """
    dedupe_since = datetime.now(tz=timezone.utc) - ALERT_DEDUPE_WINDOW
    try:
        existing = await AlertCondition.prisma().find_unique(
            where={"userId_causeKey": {"userId": user_id, "causeKey": cause_key}}
        )
        payload = SafeJson(data)
        if existing is None:
            created = await AlertCondition.prisma().create(
                data={
                    "userId": user_id,
                    "cause": cause,
                    "causeKey": cause_key,
                    "data": payload,
                }
            )
            return AlertConditionDTO.from_db(created)

        recently_alerted = (
            existing.sentAt is not None and existing.sentAt > dedupe_since
        )
        status = (
            AlertConditionStatus.DEFERRED
            if recently_alerted
            else AlertConditionStatus.PENDING
        )
        # Keep a DEFERRED row deferred until a Briefing consumes it; promoting
        # it back to PENDING would re-alert inside the 24h window.
        if existing.status is AlertConditionStatus.DEFERRED:
            status = AlertConditionStatus.DEFERRED

        updated = await AlertCondition.prisma().update(
            where={"id": existing.id},
            data={
                "cause": cause,
                "data": payload,
                "status": status,
                "resolvedAt": None,
                # A re-raise after a Briefing already reported it is news
                # again, so let the next Briefing carry it.
                "briefedAt": None,
            },
        )
        if updated is None:
            raise DatabaseError(f"Alert condition {existing.id} vanished mid-update")
        return AlertConditionDTO.from_db(updated)
    except DatabaseError:
        raise
    except Exception as e:
        raise DatabaseError(
            f"Failed to raise alert condition {cause_key} for user {user_id}: {e}"
        ) from e


async def resolve_condition(user_id: str, cause_key: str) -> bool:
    """Mark a condition fixed. Returns whether a live row was cleared.

    Called when the underlying problem goes away — including during the
    debounce window, which cancels the send outright, because an alert about a
    solved problem trains people to ignore alerts.
    """
    try:
        cleared = await AlertCondition.prisma().update_many(
            where={
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


async def get_users_with_matured_alerts(matured_before: datetime) -> list[str]:
    """Users holding at least one PENDING condition older than the debounce
    window, i.e. whose pending alerts are ready to go out as one email."""
    try:
        rows = await AlertCondition.prisma().find_many(
            where={
                "status": AlertConditionStatus.PENDING,
                "createdAt": {"lt": matured_before},
            },
            distinct=["userId"],
        )
        return [row.userId for row in rows]
    except Exception as e:
        raise DatabaseError(f"Failed to list users with matured alerts: {e}") from e


async def get_pending_conditions(user_id: str) -> list[AlertConditionDTO]:
    try:
        rows = await AlertCondition.prisma().find_many(
            where={"userId": user_id, "status": AlertConditionStatus.PENDING},
            order={"createdAt": "asc"},
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
        )
        return len(rows)
    except Exception as e:
        raise DatabaseError(
            f"Failed to count alerts sent for user {user_id}: {e}"
        ) from e


async def mark_sent(condition_ids: list[str]) -> None:
    if not condition_ids:
        return
    try:
        await AlertCondition.prisma().update_many(
            where={"id": {"in": condition_ids}},
            data={
                "status": AlertConditionStatus.SENT,
                "sentAt": datetime.now(tz=timezone.utc),
            },
        )
    except Exception as e:
        raise DatabaseError(f"Failed to mark alert conditions sent: {e}") from e


async def mark_deferred(condition_ids: list[str]) -> None:
    """Overflow past the daily cap folds into the Briefing rather than being
    dropped — nothing actionable is ever silently lost."""
    if not condition_ids:
        return
    try:
        await AlertCondition.prisma().update_many(
            where={"id": {"in": condition_ids}},
            data={"status": AlertConditionStatus.DEFERRED},
        )
    except Exception as e:
        raise DatabaseError(f"Failed to defer alert conditions: {e}") from e


async def get_briefing_conditions(user_id: str) -> list[AlertConditionDTO]:
    """Everything the next Briefing's attention block must absorb: conditions
    capped or deduped during the period, plus any still unresolved, minus the
    ones a previous Briefing already reported."""
    try:
        rows = await AlertCondition.prisma().find_many(
            where={
                "userId": user_id,
                "status": {"in": LIVE_STATUSES},
                "briefedAt": None,
            },
        )
        return [AlertConditionDTO.from_db(row) for row in rows]
    except Exception as e:
        raise DatabaseError(
            f"Failed to get briefing alert conditions for user {user_id}: {e}"
        ) from e


async def mark_briefed(condition_ids: list[str]) -> None:
    if not condition_ids:
        return
    try:
        await AlertCondition.prisma().update_many(
            where={"id": {"in": condition_ids}},
            data={"briefedAt": datetime.now(tz=timezone.utc)},
        )
    except Exception as e:
        raise DatabaseError(f"Failed to mark alert conditions briefed: {e}") from e
