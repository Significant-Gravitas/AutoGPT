"""Late execution monitoring module."""

import logging
from datetime import datetime, timedelta, timezone

from backend.data.execution import ExecutionStatus
from backend.executor import utils as execution_utils
from backend.notifications.notifications import NotificationManagerClient
from backend.util.metrics import sentry_capture_error
from backend.util.service import get_service_client
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
        self.notification_client = get_service_client(NotificationManagerClient)

    def check_late_executions(self) -> str:
        """Check for late executions and send alerts if found."""
        late_executions = execution_utils.get_db_client().get_graph_executions(
            statuses=[ExecutionStatus.QUEUED],
            created_time_gte=datetime.now(timezone.utc)
            - timedelta(
                seconds=self.config.execution_late_notification_checkrange_secs
            ),
            created_time_lte=datetime.now(timezone.utc)
            - timedelta(seconds=self.config.execution_late_notification_threshold_secs),
            limit=1000,
        )

        if not late_executions:
            return "No late executions detected."

        num_late_executions = len(late_executions)
        num_users = len(set([r.user_id for r in late_executions]))

        late_execution_details = [
            f"* `Execution ID: {exec.id}, Graph ID: {exec.graph_id}v{exec.graph_version}, User ID: {exec.user_id}, Created At: {exec.started_at.isoformat()}`"
            for exec in late_executions
        ]

        error = LateExecutionException(
            f"Late executions detected: {num_late_executions} late executions from {num_users} users "
            f"in the last {self.config.execution_late_notification_checkrange_secs} seconds. "
            f"Graph has been queued for more than {self.config.execution_late_notification_threshold_secs} seconds. "
            "Please check the executor status. Details:\n"
            + "\n".join(late_execution_details)
        )
        msg = str(error)

        sentry_capture_error(error)
        self.notification_client.discord_system_alert(msg)
        return msg


def report_late_executions() -> str:
    """Check for late executions and send Discord alerts if found."""
    monitor = LateExecutionMonitor()
    return monitor.check_late_executions()
