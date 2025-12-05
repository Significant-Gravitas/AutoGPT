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

            for graph_data in graphs:
                result = self.database_client.get_accuracy_trends_and_alerts(
                    graph_id=graph_data.graph_id,
                    user_id=graph_data.user_id,
                    days_back=21,  # 3 weeks
                    drop_threshold=self.drop_threshold,
                )

                if result.alert:
                    alert = result.alert

                    # Get graph details for better alert info
                    try:
                        graph_info = self.database_client.get_graph_metadata(
                            graph_id=alert.graph_id
                        )
                        graph_name = graph_info.name if graph_info else "Unknown Agent"
                    except Exception:
                        graph_name = "Unknown Agent"

                    # Create detailed alert message
                    alert_msg = (
                        f"🚨 **AGENT ACCURACY DROP DETECTED**\n\n"
                        f"**Agent:** {graph_name}\n"
                        f"**Graph ID:** `{alert.graph_id}`\n"
                        f"**Accuracy Drop:** {alert.drop_percent:.1f}%\n"
                        f"**Recent Performance:**\n"
                        f"  • 3-day average: {alert.three_day_avg:.1f}%\n"
                        f"  • 7-day average: {alert.seven_day_avg:.1f}%\n"
                    )

                    if alert.user_id:
                        alert_msg += f"**Owner:** {alert.user_id}\n"

                    # Send individual alert for each agent (not batched)
                    self.notification_client.discord_system_alert(
                        alert_msg, DiscordChannel.PRODUCT
                    )
                    alerts_found += 1
                    logger.warning(
                        f"Sent accuracy alert for agent: {graph_name} ({alert.graph_id})"
                    )

            if alerts_found > 0:
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
