"""Reads over delegated sub-sessions: what they cost and which are live.

A delegation has no table of its own. It is a ``ChatSession`` a
``delegate_to_expert`` call opened, recognised by its metadata, and its cost is
whatever the platform cost log recorded against that chat. Every query here is
scoped to ``user_id`` in SQL, so another user's rows are invisible rather than
filtered afterwards.
"""

from datetime import datetime

from prisma.models import Expert, User

from backend.copilot.delegation_settings import DelegationSettings
from backend.data import db
from backend.util.json import SafeJson


def delegated_session_sql(alias: str = "") -> str:
    """SQL predicate for a ``delegate_to_expert`` thread.

    Opened by another session, running as a different expert than the one that
    asked, and not handed off for good. ``run_sub_session`` subs carry the same
    provenance but stay in their spawner's scope, so the expert comparison is
    what tells the two apart.
    """
    meta, expert = f'{alias}"metadata"', f'{alias}"expertId"'
    return (
        f"({meta} ->> 'delegated_by_session_id') IS NOT NULL "
        f"AND ({meta} ->> 'handed_off_from_expert_id') IS NULL "
        f"AND {expert} IS NOT NULL "
        f"AND {expert} IS DISTINCT FROM ({meta} ->> 'delegated_by_expert_id')"
    )


async def get_session_costs(user_id: str, session_ids: list[str]) -> dict[str, int]:
    """Logged spend per chat session, in microdollars.

    Sessions with no cost rows are absent. Cost rows are written fire-and-forget
    after each turn, so a turn that finished a moment ago may not be counted yet.
    """
    if not session_ids:
        return {}
    rows = await db.query_raw_with_schema(
        'SELECT "chatSessionId" AS session_id, '
        'COALESCE(SUM("costMicrodollars"), 0)::bigint AS total '
        'FROM {schema_prefix}"PlatformCostLog" '
        'WHERE "userId" = $1 AND "chatSessionId" = ANY($2::text[]) '
        'GROUP BY "chatSessionId"',
        user_id,
        session_ids,
    )
    return {str(row["session_id"]): int(row["total"] or 0) for row in rows}


async def get_delegation_spend_since(user_id: str, since: datetime) -> int:
    """Microdollars *user_id*'s delegated threads have logged since *since*."""
    rows = await db.query_raw_with_schema(
        'SELECT COALESCE(SUM(l."costMicrodollars"), 0)::bigint AS total '
        'FROM {schema_prefix}"PlatformCostLog" l '
        'JOIN {schema_prefix}"ChatSession" s ON s."id" = l."chatSessionId" '
        'WHERE l."userId" = $1 AND s."userId" = $1 '
        # The column holds UTC without a zone; the driver binds timestamptz.
        "AND l.\"createdAt\" >= ($2::timestamptz AT TIME ZONE 'UTC') AND "
        + delegated_session_sql("s."),
        user_id,
        since,
    )
    return int(rows[0]["total"] or 0) if rows else 0


async def get_delegation_settings(user_id: str) -> DelegationSettings:
    """The user's saved settings, or the defaults if they never saved any."""
    user = await User.prisma().find_unique(where={"id": user_id})
    if user is None or user.delegationSettings is None:
        return DelegationSettings()
    return DelegationSettings.model_validate(user.delegationSettings)


async def update_delegation_settings(
    user_id: str, settings: DelegationSettings
) -> DelegationSettings:
    await User.prisma().update(
        where={"id": user_id},
        data={"delegationSettings": SafeJson(settings.model_dump())},
    )
    return settings


async def get_expert_hired_at(user_id: str, expert_id: str) -> datetime | None:
    """When *user_id* hired the expert; None if it is not theirs."""
    expert = await Expert.prisma().find_first(
        where={"id": expert_id, "ownerUserId": user_id}
    )
    return expert.createdAt if expert else None


async def raise_delegation_cap(
    session_id: str, user_id: str, by_usd: float, question_key: str
) -> float:
    """Lift a delegated thread's cap by *by_usd*, once per *question_key*;
    the cap after the call.

    ``question_key`` names the cap question being answered: a second raise for
    the same question (a retried answer) leaves the cap as it is. Writes only
    those keys, like ``set_session_pending_question``, so a turn in flight
    cannot round-trip a stale copy of the metadata over them.
    """
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatSession" SET "metadata" = '
        "COALESCE(\"metadata\", '{{}}'::jsonb) || jsonb_build_object("
        "'delegation_cap_usd', "
        "COALESCE((\"metadata\" ->> 'delegation_cap_usd')::float, 0) + $3::float, "
        "'delegation_cap_raised_for', $4::text) "
        'WHERE "id" = $1 AND "userId" = $2 '
        "AND (\"metadata\" ->> 'delegation_cap_raised_for') IS DISTINCT FROM $4",
        session_id,
        user_id,
        max(0.0, by_usd),
        question_key,
    )
    rows = await db.query_raw_with_schema(
        "SELECT (\"metadata\" ->> 'delegation_cap_usd')::float AS cap "
        'FROM {schema_prefix}"ChatSession" WHERE "id" = $1 AND "userId" = $2',
        session_id,
        user_id,
    )
    return float(rows[0]["cap"] or 0.0) if rows else 0.0


async def stop_delegation_at_cap(session_id: str, user_id: str) -> None:
    """Record that the user chose to stop a thread at its cap."""
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatSession" SET "metadata" = '
        "COALESCE(\"metadata\", '{{}}'::jsonb) || "
        "jsonb_build_object('delegation_cap_stopped', true) "
        'WHERE "id" = $1 AND "userId" = $2',
        session_id,
        user_id,
    )
