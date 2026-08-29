"""Late execution monitoring module."""

import logging
from datetime import datetime, timedelta, timezone

from backend.data.execution import ExecutionStatus
from backend.util.clients import (
    get_database_manager_client,
    get_notification_manager_client,
)
from backend.util.metrics import sentry_capture_error
from backend.util.settings import Config

logger = logging.getLogger(__name__)
config = Config()


class LateExecutionException(Exception):
    """Exception raised when late executions are detected."""

    pass


class LateExecutionMonitor:
    """Monitor late executions and send alerts when thresholds are exceeded."""

    def __init__(self):
        self.config = config
        self.notification_client = get_notification_manager_client()

    def check_late_executions(self) -> str:
        """Check for late executions and send alerts if found."""

        # Check for QUEUED executions
        queued_late_executions = get_database_manager_client().get_graph_executions(
            statuses=[ExecutionStatus.QUEUED],
            created_time_gte=datetime.now(timezone.utc)
            - timedelta(
                seconds=self.config.execution_late_notification_checkrange_secs
            ),
            created_time_lte=datetime.now(timezone.utc)
            - timedelta(seconds=self.config.execution_late_notification_threshold_secs),
            limit=1000,
        )

        # Check for RUNNING executions stuck for more than 24 hours
        running_late_executions = get_database_manager_client().get_graph_executions(
            statuses=[ExecutionStatus.RUNNING],
            created_time_gte=datetime.now(timezone.utc)
            - timedelta(hours=24)
            - timedelta(
                seconds=self.config.execution_late_notification_checkrange_secs
            ),
            created_time_lte=datetime.now(timezone.utc) - timedelta(hours=24),
            limit=1000,
        )

        all_late_executions = queued_late_executions + running_late_executions

        if not all_late_executions:
            return "No late executions detected."

        # Sort by started time (oldest first), with None values (unstarted) first
        all_late_executions.sort(
            key=lambda x: x.started_at or datetime.min.replace(tzinfo=timezone.utc)
        )

        num_total_late = len(all_late_executions)
        num_queued = len(queued_late_executions)
        num_running = len(running_late_executions)
        num_users = len(set([r.user_id for r in all_late_executions]))

        # Truncate to max entries
        tuncate_size = 5
        truncated_executions = all_late_executions[:tuncate_size]
        was_truncated = num_total_late > tuncate_size

        late_execution_details = [
            f"* `Execution ID: {exec.id}, Graph ID: {exec.graph_id}v{exec.graph_version}, User ID: {exec.user_id}, Status: {exec.status}, Started At: {exec.started_at.isoformat() if exec.started_at else 'Not started'}`"
            for exec in truncated_executions
        ]

        message_parts = [
            f"Late executions detected: {num_total_late} total late executions ({num_queued} QUEUED, {num_running} RUNNING) from {num_users} users.",
            f"QUEUED executions have been waiting for more than {self.config.execution_late_notification_threshold_secs} seconds.",
            "RUNNING executions have been running for more than 24 hours.",
            "Please check the executor status.",
        ]

        if was_truncated:
            message_parts.append(
                f"\nShowing first {tuncate_size} of {num_total_late} late executions:"
            )
        else:
            message_parts.append("\nDetails:")

        error_message = (
            "\n".join(message_parts) + "\n" + "\n".join(late_execution_details)
        )

        error = LateExecutionException(error_message)
        msg = str(error)

        sentry_capture_error(error)
        self.notification_client.discord_system_alert(msg)
        return msg


def report_late_executions() -> str:
    """Check for late executions and send Discord alerts if found."""
    monitor = LateExecutionMonitor()
    return monitor.check_late_executions()
