"""Notification processing monitoring module."""

import logging

from prisma.enums import NotificationType
from pydantic import BaseModel

from backend.util.clients import get_notification_manager_client

logger = logging.getLogger(__name__)


class NotificationJobArgs(BaseModel):
    notification_types: list[NotificationType]
    cron: str


def process_existing_batches(**kwargs):
    """Process existing notification batches."""
    args = NotificationJobArgs(**kwargs)
    try:
        logging.info(
            f"Processing existing batches for notification type {args.notification_types}"
        )
        get_notification_manager_client().process_existing_batches(
            args.notification_types
        )
    except Exception as e:
        logger.exception(f"Error processing existing batches: {e}")


def process_weekly_summary(**kwargs):
    """Process weekly summary notifications."""
    try:
        logging.info("Processing weekly summary")
        get_notification_manager_client().queue_weekly_summary()
    except Exception as e:
        logger.exception(f"Error processing weekly summary: {e}")
