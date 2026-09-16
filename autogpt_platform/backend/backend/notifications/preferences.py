"""Which notifications a user has asked for.

The volume knob, not a checkbox list: a frequency for the Briefing, a switch
for Alerts, and a switch for store verdicts. Billing and account messages are
service mail — they are about the customer's account rather than a promotion,
so they are sent regardless of these settings.
"""

from prisma.enums import NotificationType

from backend.data.notifications import NotificationPreference

# Service messages: sent even to someone who has opted out of everything else.
SERVICE_MESSAGES = frozenset(
    {
        NotificationType.SUBSCRIPTION_WELCOME,
        NotificationType.PAYMENT_FAILED,
        NotificationType.PAYMENT_FINAL_NOTICE,
        NotificationType.SUBSCRIPTION_CANCELLED,
        NotificationType.SUBSCRIPTION_RESUMED,
        NotificationType.SUBSCRIPTION_ENDED,
    }
)


def wants_notification(
    preference: NotificationPreference, notification_type: NotificationType
) -> bool:
    if notification_type in SERVICE_MESSAGES:
        return True
    if notification_type is NotificationType.BRIEFING:
        return preference.wants_briefing
    if notification_type is NotificationType.ALERT:
        return preference.alerts_enabled
    if notification_type is NotificationType.VERDICT:
        return preference.store_verdicts_enabled
    # Ops mail never reaches this path: it goes to the refunds team, not to a
    # user with preferences.
    return True
