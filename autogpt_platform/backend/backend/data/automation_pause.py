"""Pause/resume a user's automations (cron schedules + webhook-trigger presets)
when their payment lapses / is restored. Resumption only touches automations
carrying the payment-lapse marker, so anything the user paused themselves stays
off. Automations tagged with a team/external org are excluded — those are funded
by the org, not the member's personal subscription — but the user's own personal
org (which every schedule/preset is tagged with since org dual-write) counts as
personally funded.
"""

import logging

from prisma.enums import NotificationType, PresetDeactivationReason
from prisma.models import AgentPreset, OrgMember
from prisma.types import AgentPresetWhereInput
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
    personal_org_id = await _get_personal_org_id(user_id)
    # A scheduler outage must not skip trigger deactivation; the error is
    # re-raised after the preset update so the caller still alerts/retries.
    schedules = 0
    scheduler_error: Exception | None = None
    try:
        schedules = await get_scheduler_client().pause_user_graph_schedules(
            user_id=user_id,
            reason=SCHEDULE_PAUSE_REASON_PAYMENT_LAPSED,
            personal_org_id=personal_org_id,
        )
    except Exception as e:
        scheduler_error = e
        logger.error(f"Failed to pause schedules for user {user_id}: {e}")
    # deactivationReason=None keeps user-deactivated presets untouched, so a
    # repeated webhook can't overwrite a user's own deactivation.
    where: AgentPresetWhereInput = {
        "userId": user_id,
        "isActive": True,
        "isDeleted": False,
        "deactivationReason": None,
        "OR": [
            {"organizationId": None},
            {"organizationId": personal_org_id},
        ],
    }
    triggers = await AgentPreset.prisma().update_many(
        where=where,
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
    if scheduler_error is not None:
        raise scheduler_error
    return summary


async def resume_automations_after_payment_restored(
    user_id: str,
) -> AutomationPauseSummary:
    personal_org_id = await _get_personal_org_id(user_id)
    schedules = 0
    scheduler_error: Exception | None = None
    try:
        schedules = await get_scheduler_client().resume_user_graph_schedules(
            user_id=user_id,
            reason=SCHEDULE_PAUSE_REASON_PAYMENT_LAPSED,
            personal_org_id=personal_org_id,
        )
    except Exception as e:
        scheduler_error = e
        logger.error(f"Failed to resume schedules for user {user_id}: {e}")
    triggers = await AgentPreset.prisma().update_many(
        where={
            "userId": user_id,
            "isActive": False,
            "isDeleted": False,
            "deactivationReason": PresetDeactivationReason.PAYMENT_LAPSED,
            # Same personal-org predicate as pause: a preset that became
            # team-owned after being payment-lapsed stays off — it's funded by
            # the org, not the member's restored personal subscription.
            "OR": [
                {"organizationId": None},
                {"organizationId": personal_org_id},
            ],
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
    if scheduler_error is not None:
        raise scheduler_error
    return summary


async def has_payment_lapsed_automations(user_id: str) -> bool:
    """Cheap indexed check for presets still deactivated by a payment lapse.

    Lets the tier-transition hook self-heal a partially-failed resume on a
    same-tier paid retry without paying for the scheduler scan on every paid
    webhook where nothing is marked. Schedule-only stranded pauses aren't
    covered here — full recovery of those needs a persisted pending marker.
    """
    count = await AgentPreset.prisma().count(
        where={
            "userId": user_id,
            "isActive": False,
            "isDeleted": False,
            "deactivationReason": PresetDeactivationReason.PAYMENT_LAPSED,
        }
    )
    return count > 0


async def _get_personal_org_id(user_id: str) -> str | None:
    # Mirrors orgs.db._find_personal_org_member without the bootstrap side
    # effect — a billing event must not create orgs.
    member = await OrgMember.prisma().find_first(
        where={
            "userId": user_id,
            "isOwner": True,
            "Org": {"is": {"isPersonal": True, "deletedAt": None}},
        },
        order={"createdAt": "asc"},
    )
    return member.orgId if member else None


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
