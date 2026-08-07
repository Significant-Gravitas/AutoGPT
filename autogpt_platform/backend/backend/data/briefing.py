from datetime import date, datetime, time, timezone

import prisma.errors
import prisma.models
from pydantic import BaseModel

from backend.util.json import SafeJson


class BriefingRecord(BaseModel):
    id: str
    user_id: str
    briefing_date: date
    content: dict
    created_at: datetime

    @classmethod
    def from_db(cls, row: prisma.models.UserBriefing) -> "BriefingRecord":
        return cls(
            id=row.id,
            user_id=row.userId,
            briefing_date=row.briefingDate.date(),
            content=dict(row.content) if row.content else {},
            created_at=row.createdAt,
        )


def _as_db_date(briefing_date: date) -> datetime:
    return datetime.combine(briefing_date, time.min, tzinfo=timezone.utc)


async def create_briefing(
    user_id: str, briefing_date: date, content: dict
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


async def get_latest_briefing(user_id: str) -> BriefingRecord | None:
    row = await prisma.models.UserBriefing.prisma().find_first(
        where={"userId": user_id}, order={"createdAt": "desc"}
    )
    return BriefingRecord.from_db(row) if row else None
