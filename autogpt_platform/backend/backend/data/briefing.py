from datetime import date, datetime, time, timezone
from typing import Any

import prisma.errors
import prisma.models
from pydantic import BaseModel

from backend.util.json import SafeJson

# Serialized BriefingContent (see backend/copilot/briefing/models.py). Left
# untyped at this layer so a shape written by another composer version still
# round-trips; callers re-validate against the model.
BriefingContentJson = dict[str, Any]

# Upper bound on how far back get_latest_briefings looks for a readable
# briefing when the newest rows fail validation.
_LATEST_BRIEFING_SCAN_LIMIT = 5


class BriefingRecord(BaseModel):
    id: str
    user_id: str
    briefing_date: date
    content: BriefingContentJson
    created_at: datetime
    delivered_at: datetime | None = None

    @classmethod
    def from_db(cls, row: prisma.models.UserBriefing) -> "BriefingRecord":
        return cls(
            id=row.id,
            user_id=row.userId,
            briefing_date=row.briefingDate.date(),
            content=dict(row.content) if row.content else {},
            created_at=row.createdAt,
            delivered_at=row.deliveredAt,
        )


def _as_db_date(briefing_date: date) -> datetime:
    return datetime.combine(briefing_date, time.min, tzinfo=timezone.utc)


async def create_briefing(
    user_id: str, briefing_date: date, content: BriefingContentJson
) -> BriefingRecord:
    try:
        row = await prisma.models.UserBriefing.prisma().create(
            data={
                "userId": user_id,
                "briefingDate": _as_db_date(briefing_date),
                "content": SafeJson(content),
            }
        )
    except prisma.errors.UniqueViolationError:
        existing = await prisma.models.UserBriefing.prisma().find_first(
            where={"userId": user_id, "briefingDate": _as_db_date(briefing_date)}
        )
        if existing is None:
            raise
        return BriefingRecord.from_db(existing)
    return BriefingRecord.from_db(row)


async def get_briefing_for_date(
    user_id: str, briefing_date: date
) -> BriefingRecord | None:
    row = await prisma.models.UserBriefing.prisma().find_first(
        where={"userId": user_id, "briefingDate": _as_db_date(briefing_date)}
    )
    return BriefingRecord.from_db(row) if row else None


async def mark_briefing_delivered(user_id: str, briefing_id: str) -> None:
    await prisma.models.UserBriefing.prisma().update_many(
        where={"id": briefing_id, "userId": user_id},
        data={"deliveredAt": datetime.now(timezone.utc)},
    )


async def get_latest_briefings(
    user_id: str, limit: int = _LATEST_BRIEFING_SCAN_LIMIT
) -> list[BriefingRecord]:
    """Return the user's most recent briefings, newest covered date first.

    More than one is returned so a caller that re-validates stored content
    can fall back to the newest *readable* briefing instead of showing
    nothing when a single corrupt row happens to sit on the newest date.

    Ordered by briefingDate, not createdAt: a backfill or retry can write an
    earlier date's briefing after a later date's already exists, and the
    "latest" briefing means the latest date it covers, not insertion order.
    """
    rows = await prisma.models.UserBriefing.prisma().find_many(
        where={"userId": user_id}, order={"briefingDate": "desc"}, take=limit
    )
    return [BriefingRecord.from_db(row) for row in rows]
