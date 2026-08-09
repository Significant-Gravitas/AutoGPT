import logging
from datetime import date, datetime

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
    content: BriefingContent


@router.get("/latest", summary="Get Latest Briefing")
async def get_latest_briefing(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> BriefingResponse | None:
    """Return the user's most recent readable briefing, or None if there is none.

    Stored content that fails to validate against the current BriefingContent
    shape (e.g. written by an older/newer version of the composer) is skipped
    rather than 500ing the request — and rather than hiding every older
    briefing behind one unreadable row on the newest date.
    """
    for record in await briefing_db.get_latest_briefings(user_id):
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
            content=content,
        )

    return None
