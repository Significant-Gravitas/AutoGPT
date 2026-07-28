"""Pause/resume a user's automations (cron schedules + webhook-trigger presets)
when their payment lapses / is restored. Resumption only touches automations
carrying the payment-lapse marker, so anything the user paused themselves stays
off. Org/team-tagged automations are excluded — they are funded by the org, not
the member's personal subscription.
"""

import logging

from prisma.enums import NotificationType, PresetDeactivationReason
from prisma.models import AgentPreset
from pydantic import BaseModel

from backend.data.notifications import (
    AutomationsPausedData,
    AutomationsResumedData,
    NotificationEventModel,
)
from backend.notifications.notifications import queue_notification_async
from backend.util.clients import get_scheduler_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

SCHEDULE_PAUSE_REASON_PAYMENT_LAPSED = "payment_lapsed"


class AutomationPauseSummary(BaseModel):
    schedules: int = 0
    triggers: int = 0

    @property
    def total(self) -> int:
        return self.schedules + self.triggers


async def pause_automations_for_payment_lapse(user_id: str) -> AutomationPauseSummary:
    schedules = await get_scheduler_client().pause_user_graph_schedules(
        user_id=user_id, reason=SCHEDULE_PAUSE_REASON_PAYMENT_LAPSED
    )
    # deactivationReason=None keeps user-deactivated presets untouched, so a
    # repeated webhook can't overwrite a user's own deactivation.
    triggers = await AgentPreset.prisma().update_many(
        where={
            "userId": user_id,
            "isActive": True,
            "isDeleted": False,
            "organizationId": None,
            "deactivationReason": None,
        },
        data={
            "isActive": False,
            "deactivationReason": PresetDeactivationReason.PAYMENT_LAPSED,
        },
    )
    summary = AutomationPauseSummary(schedules=schedules, triggers=triggers)
    if summary.total:
        logger.info(
            f"Paused {summary.schedules} schedule(s) and {summary.triggers} "
            f"trigger(s) for user {user_id} after payment lapse"
        )
        await _notify_paused(user_id, summary)
    return summary


async def resume_automations_after_payment_restored(
    user_id: str,
) -> AutomationPauseSummary:
    schedules = await get_scheduler_client().resume_user_graph_schedules(
        user_id=user_id, reason=SCHEDULE_PAUSE_REASON_PAYMENT_LAPSED
    )
    triggers = await AgentPreset.prisma().update_many(
        where={
            "userId": user_id,
            "isActive": False,
            "isDeleted": False,
            "deactivationReason": PresetDeactivationReason.PAYMENT_LAPSED,
        },
        data={"isActive": True, "deactivationReason": None},
    )
    summary = AutomationPauseSummary(schedules=schedules, triggers=triggers)
    if summary.total:
        logger.info(
            f"Resumed {summary.schedules} schedule(s) and {summary.triggers} "
            f"trigger(s) for user {user_id} after payment restored"
        )
        await _notify_resumed(user_id, summary)
    return summary


async def _notify_paused(user_id: str, summary: AutomationPauseSummary) -> None:
    base_url = settings.config.frontend_base_url or settings.config.platform_base_url
    await queue_notification_async(
        NotificationEventModel(
            user_id=user_id,
            type=NotificationType.AUTOMATIONS_PAUSED,
            data=AutomationsPausedData(
                paused_schedules=summary.schedules,
                paused_triggers=summary.triggers,
                billing_page_link=f"{base_url}/settings/billing",
            ),
        )
    )


async def _notify_resumed(user_id: str, summary: AutomationPauseSummary) -> None:
    await queue_notification_async(
        NotificationEventModel(
            user_id=user_id,
            type=NotificationType.AUTOMATIONS_RESUMED,
            data=AutomationsResumedData(
                resumed_schedules=summary.schedules,
                resumed_triggers=summary.triggers,
            ),
        )
    )
