"""Where a user came from, captured once around signup.

The browser reports the identities the analytics tools knew the user by
before they had an account (our anonymous id, PostHog's device id, the
DataFast visitor) plus the first landing page. Each field is written at most
once: later reports only fill fields that are still empty, so a returning
user's second device never rewrites the channel that actually brought them.

Where the user came FROM is only accepted while the account is new. An older
account reporting for the first time still contributes its analytics ids, so
the join key works for everyone, but its landing page and UTM tags describe
the session it happened to report in rather than a signup, and write-once
would make that permanent.
"""

from datetime import datetime, timedelta, timezone
from typing import cast

import prisma.errors
import prisma.models
import prisma.types
from pydantic import BaseModel, Field


class UserAttributionInput(BaseModel):
    anonymous_id: str | None = Field(default=None, max_length=128)
    posthog_distinct_id: str | None = Field(default=None, max_length=128)
    datafast_visitor_id: str | None = Field(default=None, max_length=128)
    datafast_session_id: str | None = Field(default=None, max_length=128)
    landing_path: str | None = Field(default=None, max_length=2048)
    referrer: str | None = Field(default=None, max_length=2048)
    utm_source: str | None = Field(default=None, max_length=256)
    utm_medium: str | None = Field(default=None, max_length=256)
    utm_campaign: str | None = Field(default=None, max_length=256)
    signup_method: str | None = Field(default=None, max_length=32)


class UserAttribution(UserAttributionInput):
    user_id: str
    created_at: datetime

    @classmethod
    def from_db(cls, row: prisma.models.UserAttribution) -> "UserAttribution":
        return cls(
            user_id=row.userId,
            created_at=row.createdAt,
            anonymous_id=row.anonymousId,
            posthog_distinct_id=row.posthogDistinctId,
            datafast_visitor_id=row.datafastVisitorId,
            datafast_session_id=row.datafastSessionId,
            landing_path=row.landingPath,
            referrer=row.referrer,
            utm_source=row.utmSource,
            utm_medium=row.utmMedium,
            utm_campaign=row.utmCampaign,
            signup_method=row.signupMethod,
        )


_COLUMNS: dict[str, str] = {
    "anonymous_id": "anonymousId",
    "posthog_distinct_id": "posthogDistinctId",
    "datafast_visitor_id": "datafastVisitorId",
    "datafast_session_id": "datafastSessionId",
    "landing_path": "landingPath",
    "referrer": "referrer",
    "utm_source": "utmSource",
    "utm_medium": "utmMedium",
    "utm_campaign": "utmCampaign",
    "signup_method": "signupMethod",
}

# Where the user came from, as opposed to who they are. Only written while the
# account is inside SIGNUP_ATTRIBUTION_WINDOW.
_ACQUISITION_COLUMNS = frozenset(
    {
        "landingPath",
        "referrer",
        "utmSource",
        "utmMedium",
        "utmCampaign",
        "signupMethod",
    }
)

# Generous enough for a slow email verification or a retried report, far short
# of the age of any account that predates this feature.
SIGNUP_ATTRIBUTION_WINDOW = timedelta(days=7)


async def record_user_attribution(
    user_id: str, data: UserAttributionInput
) -> UserAttribution:
    """Create the row, or fill in fields that are still empty. Never overwrites."""
    provided: dict[str, str] = {
        column: value
        for field, column in _COLUMNS.items()
        if (value := getattr(data, field))
    }
    if any(column in _ACQUISITION_COLUMNS for column in provided) and not (
        await _is_recent_signup(user_id)
    ):
        provided = {
            column: value
            for column, value in provided.items()
            if column not in _ACQUISITION_COLUMNS
        }
    existing = await prisma.models.UserAttribution.prisma().find_unique(
        where={"userId": user_id}
    )
    if existing is None:
        # The columns are a closed set (see _COLUMNS); the cast only tells the
        # type checker the dynamic key set is the generated input shape.
        create_input = cast(
            prisma.types.UserAttributionCreateInput,
            {"userId": user_id, **provided},
        )
        try:
            row = await prisma.models.UserAttribution.prisma().create(data=create_input)
            return UserAttribution.from_db(row)
        except prisma.errors.UniqueViolationError:
            # Two tabs reported at once and the other one won the create;
            # fall through and fill whatever it left empty.
            existing = await prisma.models.UserAttribution.prisma().find_unique(
                where={"userId": user_id}
            )
            if existing is None:
                raise

    missing = {
        column: value
        for column, value in provided.items()
        if not getattr(existing, column)
    }
    if not missing:
        return UserAttribution.from_db(existing)
    update_input = cast(prisma.types.UserAttributionUpdateInput, missing)
    row = await prisma.models.UserAttribution.prisma().update(
        where={"userId": user_id}, data=update_input
    )
    return UserAttribution.from_db(row or existing)


async def _is_recent_signup(user_id: str) -> bool:
    user = await prisma.models.User.prisma().find_unique(where={"id": user_id})
    if user is None:
        return False
    created_at = user.createdAt
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - created_at <= SIGNUP_ATTRIBUTION_WINDOW
