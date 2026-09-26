"""Data access for the memory-scope schedule registry.

One ``MemoryScopeSchedule`` row per memory scope (the account, or one hired
expert; see ``copilot/graphiti/scope.py``) records whether the scope's dream
crons may run, the timezone they were registered in, their APScheduler job ids
and when each last ran. ``copilot/dream/registry.py`` owns the logic; this
module is only its storage. The scheduler and copilot-executor processes have
no Prisma client and reach it through ``db_accessors.memory_schedule_db()``.

Ownership: every function that names a scope also takes its owner's user id
and matches on both, so a caller only ever reaches rows of the user it names.
The two backfill listings at the bottom are unscoped on purpose and are not
exposed over the DatabaseManager RPC.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Literal

import prisma.models
import prisma.types
from prisma.enums import MemoryScopeScheduleState, ResourceVisibility
from prisma.errors import UniqueViolationError
from pydantic import BaseModel

from backend.util.exceptions import NotAuthorizedError

ScopeRunKind = Literal["nightly", "community"]


class MemoryScopeSchedule(BaseModel):
    """One memory scope's registry row."""

    scope_key: str
    user_id: str
    expert_id: str | None
    timezone: str
    state: MemoryScopeScheduleState
    community_job_id: str | None
    nightly_job_id: str | None
    last_nightly_run_at: datetime | None
    last_community_run_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_db(row: prisma.models.MemoryScopeSchedule) -> "MemoryScopeSchedule":
        return MemoryScopeSchedule(
            scope_key=row.scopeKey,
            user_id=row.userId,
            expert_id=row.expertId,
            timezone=row.timezone,
            state=row.state,
            community_job_id=row.communityJobId,
            nightly_job_id=row.nightlyJobId,
            last_nightly_run_at=row.lastNightlyRunAt,
            last_community_run_at=row.lastCommunityRunAt,
            created_at=row.createdAt,
            updated_at=row.updatedAt,
        )


class LiveExpertScope(BaseModel):
    """A hired, unarchived expert: a scope the backfill schedules or pauses."""

    expert_id: str
    user_id: str
    paused: bool


async def get_scope_schedule(
    user_id: str, scope_key: str
) -> MemoryScopeSchedule | None:
    """The scope's row, or None when it has none or it is another user's."""
    row = await prisma.models.MemoryScopeSchedule.prisma().find_first(
        where={"scopeKey": scope_key, "userId": user_id}
    )
    return MemoryScopeSchedule.from_db(row) if row else None


async def claim_scope_schedule(
    user_id: str,
    scope_key: str,
    expert_id: str | None,
    user_timezone: str,
    state: MemoryScopeScheduleState = MemoryScopeScheduleState.ACTIVE,
) -> MemoryScopeSchedule:
    """Create the scope's row in ``state`` unless one exists, and return it.

    An existing row comes back unchanged: whoever created it set its state.
    Concurrent first claims all return the one row that won: the create is
    tried, and on the unique violation the winner is read back. Raises
    ``NotAuthorizedError`` when the key is another user's.
    """
    data: prisma.types.MemoryScopeScheduleCreateInput = {
        "scopeKey": scope_key,
        "userId": user_id,
        "expertId": expert_id,
        "timezone": user_timezone,
        "state": state,
    }
    try:
        row = await prisma.models.MemoryScopeSchedule.prisma().create(data=data)
    except UniqueViolationError:
        row = await _claim_winner(data)
    if row.userId != user_id:
        raise NotAuthorizedError(f"Memory scope {scope_key[:12]} is another user's")
    return MemoryScopeSchedule.from_db(row)


async def _claim_winner(
    data: prisma.types.MemoryScopeScheduleCreateInput,
) -> prisma.models.MemoryScopeSchedule:
    """The row a concurrent claim created. If it was deleted before this
    read, the create is tried once more, and a second violation (or one on
    another key, such as a second row for the same expert) propagates."""
    table = prisma.models.MemoryScopeSchedule.prisma()
    row = await table.find_unique(where={"scopeKey": data["scopeKey"]})
    return row if row is not None else await table.create(data=data)


async def record_scope_jobs(
    user_id: str,
    scope_key: str,
    *,
    user_timezone: str,
    community_job_id: str | None,
    nightly_job_id: str | None,
) -> bool:
    """Record the scope's job ids and the timezone they run in.

    Writes only an ACTIVE row. False means the scope was paused, wiped or
    deleted meanwhile, and the caller must take down what it just registered.
    """
    updated = await prisma.models.MemoryScopeSchedule.prisma().update_many(
        where={
            "scopeKey": scope_key,
            "userId": user_id,
            "state": MemoryScopeScheduleState.ACTIVE,
        },
        data={
            "timezone": user_timezone,
            "communityJobId": community_job_id,
            "nightlyJobId": nightly_job_id,
        },
    )
    return updated > 0


async def set_scope_state(
    user_id: str,
    scope_key: str,
    state: MemoryScopeScheduleState,
    *,
    only_from: Sequence[MemoryScopeScheduleState] | None = None,
) -> bool:
    """Move the scope to ``state``; False when it has no row, or, with
    ``only_from``, when the row is in none of those states. The check and
    the write are one statement, so a concurrent change cannot slip between
    them.

    Leaving ACTIVE forgets the job ids too: the registry removes the jobs
    together with the state change.
    """
    where: prisma.types.MemoryScopeScheduleWhereInput = {
        "scopeKey": scope_key,
        "userId": user_id,
    }
    if only_from is not None:
        where["state"] = {"in": list(only_from)}
    data: prisma.types.MemoryScopeScheduleUpdateManyMutationInput = {"state": state}
    if state != MemoryScopeScheduleState.ACTIVE:
        data["communityJobId"] = None
        data["nightlyJobId"] = None
    updated = await prisma.models.MemoryScopeSchedule.prisma().update_many(
        where=where, data=data
    )
    return updated > 0


async def forget_scope_job(user_id: str, scope_key: str, job_id: str) -> bool:
    """Drop ``job_id`` from the scope's row once the job itself is gone, so
    the next ``ensure_scope_scheduled`` registers it again."""
    table = prisma.models.MemoryScopeSchedule.prisma()
    community = await table.update_many(
        where={"scopeKey": scope_key, "userId": user_id, "communityJobId": job_id},
        data={"communityJobId": None},
    )
    nightly = await table.update_many(
        where={"scopeKey": scope_key, "userId": user_id, "nightlyJobId": job_id},
        data={"nightlyJobId": None},
    )
    return community + nightly > 0


async def record_scope_run(user_id: str, scope_key: str, kind: ScopeRunKind) -> bool:
    """Stamp the time the scope's nightly or community job last ran cleanly."""
    now = datetime.now(timezone.utc)
    data: prisma.types.MemoryScopeScheduleUpdateManyMutationInput = (
        {"lastNightlyRunAt": now} if kind == "nightly" else {"lastCommunityRunAt": now}
    )
    updated = await prisma.models.MemoryScopeSchedule.prisma().update_many(
        where={"scopeKey": scope_key, "userId": user_id}, data=data
    )
    return updated > 0


async def list_user_scope_schedules(user_id: str) -> list[MemoryScopeSchedule]:
    """Every scope row the user owns: the account's and each expert's."""
    rows = await prisma.models.MemoryScopeSchedule.prisma().find_many(
        where={"userId": user_id}, order={"scopeKey": "asc"}
    )
    return [MemoryScopeSchedule.from_db(row) for row in rows]


async def list_active_scope_schedules(
    *, after: str | None = None, limit: int = 500
) -> list[MemoryScopeSchedule]:
    """One page of ACTIVE rows across all users, by scope key; pass the last
    key of a page as ``after`` for the next. Backfill only."""
    where: prisma.types.MemoryScopeScheduleWhereInput = {
        "state": MemoryScopeScheduleState.ACTIVE
    }
    if after is not None:
        where["scopeKey"] = {"gt": after}
    rows = await prisma.models.MemoryScopeSchedule.prisma().find_many(
        where=where, order={"scopeKey": "asc"}, take=limit
    )
    return [MemoryScopeSchedule.from_db(row) for row in rows]


async def list_live_expert_scopes(
    *, after: str | None = None, limit: int = 500
) -> list[LiveExpertScope]:
    """One page of hired, unarchived, owner-only experts, by expert id; pass
    the last id of a page as ``after`` for the next. Backfill only."""
    where: prisma.types.ExpertWhereInput = {
        "isTemplate": False,
        "isArchived": False,
        "ownerUserId": {"not": None},
        "visibility": ResourceVisibility.PRIVATE,
    }
    if after is not None:
        where["id"] = {"gt": after}
    rows = await prisma.models.Expert.prisma().find_many(
        where=where, order={"id": "asc"}, take=limit
    )
    return [
        LiveExpertScope(
            expert_id=row.id,
            user_id=row.ownerUserId,
            paused=row.schedulesPausedAt is not None,
        )
        for row in rows
        if row.ownerUserId is not None
    ]


async def existing_user_ids(user_ids: list[str]) -> set[str]:
    """The subset of ``user_ids`` that still have a User row. Backfill only."""
    if not user_ids:
        return set()
    rows = await prisma.models.User.prisma().find_many(where={"id": {"in": user_ids}})
    return {row.id for row in rows}
