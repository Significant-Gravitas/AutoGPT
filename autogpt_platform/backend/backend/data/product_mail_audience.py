"""Who product mail (Briefings, Alerts, Verdicts) may reach at all, before a
user's own preferences are asked.

Set by USER_NOTIFICATIONS_AUDIENCE and USER_NOTIFICATIONS_ALLOWLIST. On the
default, subscribers: an account on a plan or trial, or whose plan or trial
ended within the lapse window. A set allowlist decides on its own, so a canary
reaches only the team inboxes on it. Subscription and trial mail is service
mail and never asks.

The rule is written twice, once for one user the consumer already has in hand
and once as a where clause for the hourly briefing pass. Keep them together.
"""

from datetime import datetime, timedelta

from prisma.enums import SubscriptionTier
from prisma.models import User
from prisma.types import UserWhereInput
from pydantic import BaseModel

from backend.util.exceptions import DatabaseError
from backend.util.settings import Settings, UserNotificationsAudience

settings = Settings()

# How long product mail keeps reaching an account after its plan or trial
# ends: the win-back window.
PRODUCT_MAIL_LAPSE_WINDOW = timedelta(days=7)


class ProductMailStanding(BaseModel):
    """The fields the audience rule reads. Narrow, because it crosses the
    DatabaseManager RPC boundary for every product notification."""

    email: str
    subscription_tier: SubscriptionTier
    subscription_ended_at: datetime | None
    trial_ends_at: datetime | None


def is_product_mail_recipient(standing: ProductMailStanding, now: datetime) -> bool:
    allowlist = product_mail_allowlist()
    if allowlist is not None:
        return standing.email.strip().lower() in allowlist
    if (
        settings.config.user_notifications_audience
        == UserNotificationsAudience.EVERYONE
    ):
        return True
    since = now - PRODUCT_MAIL_LAPSE_WINDOW
    if standing.subscription_tier != SubscriptionTier.NO_TIER:
        return True
    if standing.subscription_ended_at and standing.subscription_ended_at >= since:
        return True
    # Whatever the trial's status: payment failed, cancelled or ended.
    return bool(standing.trial_ends_at and since <= standing.trial_ends_at <= now)


def product_mail_audience_where(now: datetime) -> UserWhereInput | None:
    """`is_product_mail_recipient` as a where clause; None when everyone may."""
    allowlist = product_mail_allowlist()
    if allowlist is not None:
        if not allowlist:
            return {"id": {"in": []}}
        return {
            "OR": [
                {"email": {"equals": address, "mode": "insensitive"}}
                for address in allowlist
            ]
        }
    if (
        settings.config.user_notifications_audience
        == UserNotificationsAudience.EVERYONE
    ):
        return None
    since = now - PRODUCT_MAIL_LAPSE_WINDOW
    return {
        "OR": [
            {"subscriptionTier": {"not": SubscriptionTier.NO_TIER}},
            {"subscriptionEndedAt": {"gte": since}},
            {"subscriptionTrial": {"is": {"endsAt": {"gte": since, "lte": now}}}},
        ]
    }


def product_mail_allowlist() -> list[str] | None:
    """USER_NOTIFICATIONS_ALLOWLIST as lower-cased addresses; None when unset.

    A value that is set but names no address (a stray comma) is an empty list,
    which lets no one through rather than falling back to the audience."""
    raw = settings.config.user_notifications_allowlist
    if not raw.strip():
        return None
    return [a.strip().lower() for a in raw.split(",") if a.strip()]


async def get_product_mail_standing(user_id: str) -> ProductMailStanding | None:
    try:
        user = await User.prisma().find_unique(
            where={"id": user_id}, include={"subscriptionTrial": True}
        )
    except Exception as e:
        raise DatabaseError(
            f"Failed to read the product mail standing of user {user_id}: {e}"
        ) from e
    if user is None:
        return None
    return ProductMailStanding(
        email=user.email,
        subscription_tier=user.subscriptionTier or SubscriptionTier.NO_TIER,
        subscription_ended_at=user.subscriptionEndedAt,
        trial_ends_at=user.subscriptionTrial.endsAt if user.subscriptionTrial else None,
    )
