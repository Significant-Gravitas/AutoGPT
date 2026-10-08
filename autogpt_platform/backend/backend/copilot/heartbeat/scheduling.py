"""Keeping a user's heartbeat job in step with their settings.

Called whenever the settings are saved; unlike the briefing there is no lazy
path on every chat turn, because nothing but a save changes the job.
"""

import logging

from .config import HeartbeatConfig, checklist_is_empty

logger = logging.getLogger(__name__)


def wants_schedule(config: HeartbeatConfig) -> bool:
    """A job only for a switched-on heartbeat with something to check."""
    return config.enabled and not checklist_is_empty(config.checklist)


async def sync_heartbeat_schedule(user_id: str, config: HeartbeatConfig) -> bool:
    """Register, refresh or remove the job. False when the scheduler could
    not be reached; the saved settings stand either way, and the next save
    (or a beat that finds the heartbeat off) puts the job right."""
    from backend.util.clients import get_scheduler_client

    try:
        client = get_scheduler_client()
        if wants_schedule(config):
            await client.add_copilot_heartbeat_schedule(
                user_id=user_id, interval_minutes=config.interval_minutes
            )
        else:
            await client.remove_copilot_heartbeat_schedule(user_id=user_id)
        return True
    except Exception:
        logger.warning(
            "Heartbeat: could not sync the schedule for user %s",
            user_id[:12],
            exc_info=True,
        )
        return False
