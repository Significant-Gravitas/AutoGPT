"""Stop the emails a closed account would otherwise keep getting."""

import logging

from prisma.enums import BriefingFrequency

from backend.data.notifications import NotificationPreferenceDTO
from backend.data.user import (
    get_user_notification_preference,
    record_marketing_opt_out_by_email,
    unsubscribe_from_marketing,
    update_user_notification_preference,
)

logger = logging.getLogger(__name__)

# User.marketingOptOutSource is a free string ("signup", "email_unsubscribe").
MARKETING_OPT_OUT_SOURCE_ACCOUNT_DELETED = "account_deleted"


async def stop_account_emails(user_id: str) -> None:
    """Switch off the account's product email and opt it out of marketing.

    Product email: the Briefing, Alerts and store Verdicts go off, and the
    daily limit drops to 0, through the same preference write as the settings
    page and the one-click unsubscribe. Billing and account mail is service
    mail that ignores these settings (notifications/preferences.py).

    Marketing: the opt-out is recorded with source "account_deleted" (an
    earlier refusal keeps its own date and source), which keeps the account
    out of every later MailerLite write, and an unsubscribe is queued so
    MailerLite stops mailing a subscriber it already has.

    Idempotent. An unknown account raises NotFoundError and database errors
    propagate; a failure to queue the MailerLite unsubscribe is logged and
    does not raise, since the stored opt-out already blocks every other
    MailerLite write.
    """
    current = await get_user_notification_preference(user_id)
    await update_user_notification_preference(
        user_id,
        NotificationPreferenceDTO(
            email=current.email,
            briefing_frequency=BriefingFrequency.OFF,
            alerts_enabled=False,
            store_verdicts_enabled=False,
            daily_limit=0,
        ),
    )
    await record_marketing_opt_out_by_email(
        current.email, MARKETING_OPT_OUT_SOURCE_ACCOUNT_DELETED
    )
    await unsubscribe_from_marketing(user_id, current.email)
    logger.info(f"Account deletion stopped product and marketing email for {user_id}")
