"""Block error rate monitoring module."""

import logging
import re
from datetime import datetime, timedelta, timezone

from backend.executor import utils as execution_utils
from backend.notifications.notifications import NotificationManagerClient
from backend.util.service import get_service_client
from backend.util.settings import Config

logger = logging.getLogger(__name__)
config = Config()


class BlockErrorMonitor:
    """Monitor block error rates and send alerts when thresholds are exceeded."""
    
    def __init__(self):
        self.config = config
        self.notification_client = get_service_client(NotificationManagerClient)
    
    def check_block_error_rates(self) -> str:
        """Check block error rates and send Discord alerts if thresholds are exceeded."""
        try:
            logger.info("Checking block error rates")

            # Get executions from the last 24 hours
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=24)

            executions = execution_utils.get_db_client().get_node_executions(
                created_time_gte=start_time, created_time_lte=end_time
            )

            # Calculate error rates by block and collect error samples
            block_stats = {}
            for execution in executions:
                block_name = (
                    execution.agentNode.agentBlock.name
                    if execution.agentNode and execution.agentNode.agentBlock
                    else "Unknown"
                )

                if block_name not in block_stats:
                    block_stats[block_name] = {"total": 0, "failed": 0, "error_samples": []}

                block_stats[block_name]["total"] += 1
                if execution.executionStatus == "FAILED":
                    block_stats[block_name]["failed"] += 1

                    # Collect error samples (limit to 5 per block)
                    if len(block_stats[block_name]["error_samples"]) < 5:
                        error_message = self._extract_error_message(execution)
                        if error_message:
                            masked_error = self._mask_sensitive_data(error_message)
                            block_stats[block_name]["error_samples"].append(masked_error)

            # Check thresholds and send alerts
            threshold = self.config.block_error_rate_threshold
            alerts = []

            for block_name, stats in block_stats.items():
                if stats["total"] >= 10:  # Only check blocks with at least 10 executions
                    error_rate = stats["failed"] / stats["total"]
                    if error_rate >= threshold:
                        error_percentage = error_rate * 100

                        # Group similar errors
                        error_groups = self._group_similar_errors(stats["error_samples"])

                        alert_msg = (
                            f"🚨 Block '{block_name}' has {error_percentage:.1f}% error rate "
                            f"({stats['failed']}/{stats['total']}) in the last 24 hours"
                        )

                        if error_groups:
                            alert_msg += "\n\n📊 Error Types:"
                            for error_pattern, count in error_groups.items():
                                alert_msg += f"\n• {error_pattern} ({count}x)"

                        alerts.append(alert_msg)

            if alerts:
                msg = "Block Error Rate Alert:\n\n" + "\n\n".join(alerts)
                self.notification_client.discord_system_alert(msg)
                logger.info(f"Sent block error rate alert for {len(alerts)} blocks")
                return f"Alert sent for {len(alerts)} blocks with high error rates"
            else:
                logger.info("No blocks exceeded error rate threshold")
                return "No blocks exceeded error rate threshold"

        except Exception as e:
            logger.exception(f"Error checking block error rates: {e}")
            from backend.util.metrics import sentry_capture_error
            from backend.executor.scheduler import LateExecutionException
            
            error = LateExecutionException(f"Error checking block error rates: {e}")
            msg = str(error)
            sentry_capture_error(error)
            self.notification_client.discord_system_alert(msg)
            return msg

    def _extract_error_message(self, execution):
        """Extract error message from execution stats."""
        try:
            if hasattr(execution, "stats") and execution.stats:
                stats = execution.stats
                if isinstance(stats, dict):
                    # Look for error message in various common locations
                    error_msg = (
                        stats.get("error_message")
                        or stats.get("error")
                        or stats.get("exception")
                        or str(stats.get("output", ""))
                    )
                    return error_msg if error_msg else None
                elif isinstance(stats, str):
                    return stats
            return None
        except Exception:
            return None

    def _mask_sensitive_data(self, error_message):
        """Mask sensitive data in error messages to enable grouping."""
        if not error_message:
            return ""

        # Convert to string if not already
        error_str = str(error_message)

        # Mask numbers (replace with X)
        error_str = re.sub(r"\d+", "X", error_str)

        # Mask all caps words (likely constants/IDs)
        error_str = re.sub(r"\b[A-Z_]{3,}\b", "MASKED", error_str)

        # Mask words with underscores (likely internal variables)
        error_str = re.sub(r"\b\w*_\w*\b", "MASKED", error_str)

        # Mask UUIDs and long alphanumeric strings
        error_str = re.sub(
            r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",
            "UUID",
            error_str,
        )
        error_str = re.sub(r"\b[a-f0-9]{20,}\b", "HASH", error_str)

        # Mask file paths
        error_str = re.sub(r"(/[^/\s]+)+", "/MASKED/path", error_str)

        # Mask URLs
        error_str = re.sub(r"https?://[^\s]+", "URL", error_str)

        # Mask email addresses
        error_str = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "EMAIL", error_str
        )

        # Truncate if too long
        if len(error_str) > 100:
            error_str = error_str[:97] + "..."

        return error_str.strip()

    def _group_similar_errors(self, error_samples):
        """Group similar error messages and return counts."""
        if not error_samples:
            return {}

        error_groups = {}
        for error in error_samples:
            if error in error_groups:
                error_groups[error] += 1
            else:
                error_groups[error] = 1

        # Sort by frequency, most common first
        return dict(sorted(error_groups.items(), key=lambda x: x[1], reverse=True))