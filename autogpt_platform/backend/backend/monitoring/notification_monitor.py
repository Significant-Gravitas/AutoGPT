"""Scheduled passes for the email system.

Two jobs, matching the two families that are not purely event-driven: the
Alert's debounce window has to be emptied on a timer, and the Briefing has to
catch each user's local morning.
"""

import logging

from backend.util.clients import get_notification_manager_client

logger = logging.getLogger(__name__)


def flush_matured_alerts() -> None:
    """Send everything that has sat out the ten-minute debounce window."""
    try:
        get_notification_manager_client().flush_matured_alerts()
    except Exception as e:
        logger.exception(f"Error flushing matured alerts: {e}")


def send_due_briefings() -> None:
    """Assemble briefings for the users whose local ~07:30 this hour is."""
    try:
        get_notification_manager_client().send_due_briefings()
    except Exception as e:
        logger.exception(f"Error sending due briefings: {e}")
