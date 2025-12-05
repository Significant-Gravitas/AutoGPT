"""Execution accuracy monitoring module."""

import logging

from backend.util.clients import (
    get_database_manager_client,
    get_notification_manager_client,
)
from backend.util.metrics import DiscordChannel, sentry_capture_error
from backend.util.settings import Config

logger = logging.getLogger(__name__)
config = Config()


class AccuracyMonitor:
    """Monitor execution accuracy trends and send alerts for drops."""

    def __init__(self, drop_threshold: float = 10.0):
        self.config = config
        self.notification_client = get_notification_manager_client()
        self.database_client = get_database_manager_client()
        self.drop_threshold = drop_threshold

    def check_execution_accuracy_alerts(self) -> str:
        """Check marketplace agents for accuracy drops and send alerts."""
        try:
            logger.info("Checking execution accuracy for marketplace agents")

            # Get marketplace graphs using database client
            graphs = self.database_client.get_marketplace_graphs_for_monitoring(
                days_back=30, min_executions=10
            )

            alerts_found = 0
            alert_messages = []

            for graph_data in graphs:
                result = self.database_client.get_accuracy_trends_and_alerts(
                    graph_id=graph_data.graph_id,
                    user_id=graph_data.user_id,
                    days_back=21,  # 3 weeks
                    drop_threshold=self.drop_threshold,
                )

                if result.alert:
                    alert = result.alert

                    alert_msg = (
                        f"🚨 ACCURACY ALERT: Graph {alert.graph_id[:8]}... "
                        f"has {alert.drop_percent:.1f}% accuracy drop.\n"
                        f"3-day avg: {alert.three_day_avg:.2f}, "
                        f"7-day avg: {alert.seven_day_avg:.2f}"
                    )

                    if alert.user_id:
                        alert_msg += f"\nUser: {alert.user_id[:8]}..."

                    alert_messages.append(alert_msg)
                    alerts_found += 1

            if alert_messages:
                full_message = "Execution Accuracy Alerts:\n\n" + "\n\n".join(
                    alert_messages
                )
                self.notification_client.discord_system_alert(
                    full_message, DiscordChannel.PRODUCT
                )
                logger.warning(f"Sent accuracy alerts for {alerts_found} agents")
                return f"Alert sent for {alerts_found} agents with accuracy drops"

            logger.info("No execution accuracy alerts detected")
            return "No accuracy alerts detected"

        except Exception as e:
            logger.exception(f"Error checking execution accuracy alerts: {e}")

            error = Exception(f"Error checking execution accuracy alerts: {e}")
            msg = str(error)
            sentry_capture_error(error)
            self.notification_client.discord_system_alert(msg, DiscordChannel.PRODUCT)
            return msg


def report_execution_accuracy_alerts(drop_threshold: float = 10.0) -> str:
    """
    Check execution accuracy and send alerts if drops are detected.

    Args:
        drop_threshold: Percentage drop threshold to trigger alerts (default 10.0%)

    Returns:
        Status message indicating results of the check
    """
    monitor = AccuracyMonitor(drop_threshold=drop_threshold)
    return monitor.check_execution_accuracy_alerts()
