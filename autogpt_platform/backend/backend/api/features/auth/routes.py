import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated

from autogpt_libs.auth import get_user_id, requires_user
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi import APIRouter, Body, HTTPException, Query, Response, Security
from prisma.enums import BriefingFrequency

from backend.api.model import TimezoneResponse, UpdateTimezoneRequest
from backend.data.notifications import NotificationPreference, NotificationPreferenceDTO
from backend.data.user import (
    get_or_create_user,
    get_or_create_user_with_status,
    get_user_notification_preference,
    update_user_email,
    update_user_notification_preference,
    update_user_timezone,
    verify_preference_token,
)
from backend.util.settings import Settings

settings = Settings()
logger = logging.getLogger(__name__)

# Nothing is hoisted onto this router, tags included. Six of the seven routes
# take Security(requires_user); POST /auth/user/preferences/from-email takes
# none — it is reached from an email link and verifies its own signed token —
# so a router-level dependency would silently authenticate it.
router = APIRouter()


_tally_background_tasks: set[asyncio.Task] = set()
USER_CREATED_HEADER = "X-AutoGPT-User-Created"


@router.post(
    "/auth/user",
    summary="Get or create user",
    tags=["auth"],
    responses={
        200: {
            "description": "Successful Response",
            "headers": {
                USER_CREATED_HEADER: {
                    "description": "Whether this request created a new user",
                    "schema": {"type": "string", "enum": ["true", "false"]},
                }
            },
        }
    },
    dependencies=[Security(requires_user)],
)
async def get_or_create_user_route(
    response: Response, user_data: dict = Security(get_jwt_payload)
):
    result = await get_or_create_user_with_status(user_data)
    response.headers[USER_CREATED_HEADER] = str(result.was_created).lower()
    user = result.user

    # Fire-and-forget: populate business understanding from Tally form.
    age_seconds = (datetime.now(timezone.utc) - user.created_at).total_seconds()
    if age_seconds < 30:
        try:
            from backend.data.tally import populate_understanding_from_tally

            task = asyncio.create_task(
                populate_understanding_from_tally(user.id, user.email)
            )
            _tally_background_tasks.add(task)
            task.add_done_callback(_tally_background_tasks.discard)
        except Exception:
            logger.debug("Failed to start Tally population task", exc_info=True)

    return user.model_dump()


@router.post(
    "/auth/user/email",
    summary="Update user email",
    tags=["auth"],
    dependencies=[Security(requires_user)],
)
async def update_user_email_route(
    user_id: Annotated[str, Security(get_user_id)], email: str = Body(...)
) -> dict[str, str]:
    await update_user_email(user_id, email)

    return {"email": email}


@router.get(
    "/auth/user/timezone",
    summary="Get user timezone",
    tags=["auth"],
    dependencies=[Security(requires_user)],
)
async def get_user_timezone_route(
    user_data: dict = Security(get_jwt_payload),
) -> TimezoneResponse:
    """Get user timezone setting."""
    user = await get_or_create_user(user_data)
    return TimezoneResponse(timezone=user.timezone)


@router.post(
    "/auth/user/timezone",
    summary="Update user timezone",
    tags=["auth"],
    dependencies=[Security(requires_user)],
)
async def update_user_timezone_route(
    user_id: Annotated[str, Security(get_user_id)], request: UpdateTimezoneRequest
) -> TimezoneResponse:
    """Update user timezone. The timezone should be a valid IANA timezone identifier."""
    user = await update_user_timezone(user_id, str(request.timezone))
    return TimezoneResponse(timezone=user.timezone)


@router.get(
    "/auth/user/preferences",
    summary="Get notification preferences",
    tags=["auth"],
    dependencies=[Security(requires_user)],
)
async def get_preferences(
    user_id: Annotated[str, Security(get_user_id)],
) -> NotificationPreference:
    preferences = await get_user_notification_preference(user_id)
    return preferences


@router.post(
    "/auth/user/preferences",
    summary="Update notification preferences",
    tags=["auth"],
    dependencies=[Security(requires_user)],
)
async def update_preferences(
    user_id: Annotated[str, Security(get_user_id)],
    preferences: NotificationPreferenceDTO = Body(...),
) -> NotificationPreference:
    output = await update_user_notification_preference(user_id, preferences)
    return output


@router.post(
    "/auth/user/preferences/from-email",
    summary="Apply a volume-knob choice from a Briefing footer link",
    tags=["auth"],
)
async def apply_email_preference_choice(
    choice: Annotated[str, Query()],
    token: Annotated[str, Query()],
) -> NotificationPreference:
    """Apply one footer choice, authorised by the token in the link.

    Deliberately not `Security(requires_user)`. The session is the wrong
    authority here: the settings page applies this on arrival, so a
    session-authenticated write would let any third party change a logged-in
    reader's preferences just by getting them to follow a link. The HMAC binds
    the choice to the recipient we sent it to, exactly as the unsubscribe link
    does, and works whether or not they happen to be signed in.
    """
    user_id = verify_preference_token(token, choice)
    if user_id is None:
        raise HTTPException(status_code=400, detail="Invalid or expired link")

    current = await get_user_notification_preference(user_id)
    updated = _preference_with_choice(current, choice)
    if updated is None:
        raise HTTPException(status_code=400, detail="Unknown preference choice")
    return await update_user_notification_preference(user_id, updated)


def _preference_with_choice(
    current: NotificationPreference, choice: str
) -> NotificationPreferenceDTO | None:
    """The volume knob, server-side. "alerts" and "off" both stop the digest;
    they differ in whether alerts survive."""
    mapping: dict[str, tuple[BriefingFrequency, bool]] = {
        "daily": (BriefingFrequency.DAILY, current.alerts_enabled),
        "weekly": (BriefingFrequency.WEEKLY, current.alerts_enabled),
        "monthly": (BriefingFrequency.MONTHLY, current.alerts_enabled),
        "alerts": (BriefingFrequency.OFF, True),
        "off": (BriefingFrequency.OFF, False),
    }
    if choice not in mapping:
        return None
    frequency, alerts = mapping[choice]
    return NotificationPreferenceDTO(
        email=current.email,
        briefing_frequency=frequency,
        alerts_enabled=alerts,
        store_verdicts_enabled=current.store_verdicts_enabled,
        daily_limit=current.daily_limit,
    )
