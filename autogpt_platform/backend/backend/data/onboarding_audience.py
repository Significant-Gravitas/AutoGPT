"""Queue the onboarding role for the account's MailerLite subscriber.

The wizard's role is answered before the paywall, so most people are not in
MailerLite yet: they get their role with the checkout event
(`checkout_audience`), which reads the kept pick. This only fills in someone
who already opened checkout, such as a person who went back and changed
their pick; it never creates a subscriber, since only checkout openers belong
in MailerLite (`mailerlite.record_onboarding_profile`).

Nothing is queued for an account that may not be in MailerLite at all
(`notifications/consent.py`), and nothing here ever raises: the profile is
saved whatever happens to the MailerLite bookkeeping.
"""

import logging

from backend.data.notifications import AudienceAction
from backend.data.onboarding_role import OnboardingRole
from backend.data.user import get_user_by_id
from backend.notifications.audience_enrichment import role_fields
from backend.notifications.consent import audience_change_allowed
from backend.notifications.subscriber_fields import queue_fields

logger = logging.getLogger(__name__)


async def queue_onboarding_role(user_id: str, role: OnboardingRole) -> None:
    try:
        user = await get_user_by_id(user_id)
        if not audience_change_allowed(user, AudienceAction.ONBOARDING_PROFILE):
            return
        await queue_fields(
            user_id, user.email, role_fields(role), AudienceAction.ONBOARDING_PROFILE
        )
    except Exception:
        logger.exception(f"Could not queue the MailerLite role for {user_id}")
