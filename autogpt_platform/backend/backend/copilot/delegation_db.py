"""Reads over delegated sub-sessions: what they cost and which are live.

A delegation has no table of its own. It is a ``ChatSession`` a
``delegate_to_expert`` call opened, recognised by its metadata, and its cost is
whatever the platform cost log recorded against that chat. Every query here is
scoped to ``user_id`` in SQL, so another user's rows are invisible rather than
filtered afterwards.
"""

from backend.data import db


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
