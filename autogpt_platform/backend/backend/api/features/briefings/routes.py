import logging
from datetime import date, datetime, timedelta, timezone

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Security
from pydantic import BaseModel, ValidationError

from backend.copilot.briefing.models import BriefingContent
from backend.data import briefing as briefing_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/briefings",
    tags=["briefings"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


class BriefingResponse(BaseModel):
    id: str
    briefing_date: date
    created_at: datetime
    delivered_at: datetime | None
    content: BriefingContent


# How stale a briefing may be and still be shown as "your briefing". Beyond
# this the home card would present weeks-old runs — with deep links into
# executions long since reviewed or deleted — under a year-less "August 7"
# label. Yesterday is included so an early-morning visit still has one.
_MAX_BRIEFING_AGE_DAYS = 1


@router.get("/latest", summary="Get Latest Briefing")
async def get_latest_briefing(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> BriefingResponse | None:
    """Return the user's current briefing, or None if there isn't a fresh one.

    Only briefings covering today or yesterday qualify; anything older is
    history, not "this morning".

    Stored content that fails to validate against the current BriefingContent
    shape (e.g. written by an older/newer version of the composer) is skipped
    rather than 500ing the request — and rather than hiding every older
    briefing behind one unreadable row on the newest date.
    """
    oldest_allowed = datetime.now(timezone.utc).date() - timedelta(
        days=_MAX_BRIEFING_AGE_DAYS
    )
    for record in await briefing_db.get_latest_briefings(user_id):
        if record.briefing_date < oldest_allowed:
            break

        try:
            content = BriefingContent.model_validate(record.content)
        except ValidationError:
            logger.warning(
                "Briefing %s for user %s failed to validate against "
                "BriefingContent; falling back to the previous briefing",
                record.id,
                user_id,
            )
            continue

        return BriefingResponse(
            id=record.id,
            briefing_date=record.briefing_date,
            created_at=record.created_at,
            delivered_at=record.delivered_at,
            content=content,
        )

    return None
