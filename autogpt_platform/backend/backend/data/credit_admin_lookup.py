"""Account details for the admin credit views.

The credit ledger has no relation to `User`: its rows are billing records and
outlive a deleted account (SECRT-2854). So the admin views look emails up by
user ID, and match a search on email or name in SQL, instead of joining
through a relation; a row whose account is gone has no email.
"""

from collections.abc import Iterable

from prisma.enums import CreditTransactionType
from prisma.models import User

from backend.data.db import query_raw_with_schema

_LOOKUP_CHUNK = 1000

# $1 = the ILIKE pattern, $2 = include inactive rows, $3 = the type or NULL.
_MATCHES = """
      FROM {schema_prefix}"CreditTransaction" ct
      LEFT JOIN {schema_prefix}"User" u ON u."id" = ct."userId"
     WHERE ($2::boolean OR ct."isActive")
       AND ($3::text IS NULL OR ct."type"::text = $3::text)
       AND (ct."userId" ILIKE $1 OR u."email" ILIKE $1 OR u."name" ILIKE $1)
"""
_PAGE = f"""
    SELECT ct."transactionKey" AS key, ct."userId" AS user_id {_MATCHES}
     ORDER BY ct."createdAt" DESC
     OFFSET $4 LIMIT $5
"""
_COUNT = f"SELECT COUNT(*)::int AS n {_MATCHES}"


async def user_emails(user_ids: Iterable[str]) -> dict[str, str]:
    """Emails of the accounts that still exist, by user ID."""
    ids = list(dict.fromkeys(user_ids))
    emails: dict[str, str] = {}
    for start in range(0, len(ids), _LOOKUP_CHUNK):
        users = await User.prisma().find_many(
            where={"id": {"in": ids[start : start + _LOOKUP_CHUNK]}}
        )
        emails.update({user.id: user.email for user in users})
    return emails


async def search_history_keys(
    search: str,
    *,
    include_inactive: bool,
    transaction_filter: CreditTransactionType | None,
    skip: int,
    take: int,
) -> tuple[list[tuple[str, str]], int]:
    """One page of the ledger rows whose user ID, email or name contains
    `search` (case-insensitive), newest first, as (transaction key, user ID)
    pairs, and how many match in all."""
    args = (
        f"%{_escape_like(search)}%",
        include_inactive,
        transaction_filter.value if transaction_filter else None,
    )
    rows = await query_raw_with_schema(_PAGE, *args, skip, take)
    counted = await query_raw_with_schema(_COUNT, *args)
    total = int(counted[0]["n"]) if counted else 0
    return [(row["key"], row["user_id"]) for row in rows], total


def _escape_like(value: str) -> str:
    """Match the text literally, as Prisma's `contains` did."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
