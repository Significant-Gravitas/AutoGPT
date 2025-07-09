"""Block error rate monitoring module."""

import logging
import re
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from backend.data.block import get_block
from backend.data.execution import ExecutionStatus, NodeExecutionResult
from backend.executor import utils as execution_utils
from backend.notifications.notifications import NotificationManagerClient
from backend.util.metrics import sentry_capture_error
from backend.util.service import get_service_client
from backend.util.settings import Config

logger = logging.getLogger(__name__)
config = Config()


class BlockStatsWithSamples(BaseModel):
    """Enhanced block stats with error samples."""

    block_id: str
    block_name: str
    total_executions: int
    failed_executions: int
    error_samples: list[str] = []

    @property
    def error_rate(self) -> float:
        """Calculate error rate as a percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.failed_executions / self.total_executions) * 100


class BlockErrorMonitor:
    """Monitor block error rates and send alerts when thresholds are exceeded."""

    def __init__(self, include_top_blocks: int | None = None):
        self.config = config
        self.notification_client = get_service_client(NotificationManagerClient)
        self.include_top_blocks = (
            include_top_blocks
            if include_top_blocks is not None
            else config.block_error_include_top_blocks
        )

    def check_block_error_rates(self) -> str:
        """Check block error rates and send Discord alerts if thresholds are exceeded."""
        try:
            logger.info("Checking block error rates")

            # Get executions from the last 24 hours
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=24)

            # Use SQL aggregation to efficiently count totals and failures by block
            block_stats = self._get_block_stats_from_db(start_time, end_time)

            # For blocks with high error rates, fetch error samples
            threshold = self.config.block_error_rate_threshold
            for block_name, stats in block_stats.items():
                if stats.total_executions >= 10 and stats.error_rate >= threshold * 100:
                    # Only fetch error samples for blocks that exceed threshold
                    error_samples = self._get_error_samples_for_block(
                        stats.block_id, start_time, end_time, limit=3
                    )
                    stats.error_samples = error_samples

            # Check thresholds and send alerts
            critical_alerts = self._generate_critical_alerts(block_stats, threshold)

            if critical_alerts:
                msg = "Block Error Rate Alert:\n\n" + "\n\n".join(critical_alerts)
                self.notification_client.discord_system_alert(msg)
                logger.info(
                    f"Sent block error rate alert for {len(critical_alerts)} blocks"
                )
                return f"Alert sent for {len(critical_alerts)} blocks with high error rates"

            # If no critical alerts, check if we should show top blocks
            if self.include_top_blocks > 0:
                top_blocks_msg = self._generate_top_blocks_alert(
                    block_stats, start_time, end_time
                )
                if top_blocks_msg:
                    self.notification_client.discord_system_alert(top_blocks_msg)
                    logger.info("Sent top blocks summary")
                    return "Sent top blocks summary"

            logger.info("No blocks exceeded error rate threshold")
            return "No errors reported for today"

        except Exception as e:
            logger.exception(f"Error checking block error rates: {e}")

            error = Exception(f"Error checking block error rates: {e}")
            msg = str(error)
            sentry_capture_error(error)
            self.notification_client.discord_system_alert(msg)
            return msg

    def _get_block_stats_from_db(
        self, start_time: datetime, end_time: datetime
    ) -> dict[str, BlockStatsWithSamples]:
        """Get block execution stats using efficient SQL aggregation."""

        result = execution_utils.get_db_client().get_block_error_stats(
            start_time, end_time
        )

        block_stats = {}
        for stats in result:
            block_name = b.name if (b := get_block(stats.block_id)) else "Unknown"

            block_stats[block_name] = BlockStatsWithSamples(
                block_id=stats.block_id,
                block_name=block_name,
                total_executions=stats.total_executions,
                failed_executions=stats.failed_executions,
                error_samples=[],
            )

        return block_stats

    def _generate_critical_alerts(
        self, block_stats: dict[str, BlockStatsWithSamples], threshold: float
    ) -> list[str]:
        """Generate alerts for blocks that exceed the error rate threshold."""
        alerts = []

        for block_name, stats in block_stats.items():
            if stats.total_executions >= 10 and stats.error_rate >= threshold * 100:
                error_groups = self._group_similar_errors(stats.error_samples)

                alert_msg = (
                    f"🚨 Block '{block_name}' has {stats.error_rate:.1f}% error rate "
                    f"({stats.failed_executions}/{stats.total_executions}) in the last 24 hours"
                )

                if error_groups:
                    alert_msg += "\n\n📊 Error Types:"
                    for error_pattern, count in error_groups.items():
                        alert_msg += f"\n• {error_pattern} ({count}x)"

                alerts.append(alert_msg)

        return alerts

    def _generate_top_blocks_alert(
        self,
        block_stats: dict[str, BlockStatsWithSamples],
        start_time: datetime,
        end_time: datetime,
    ) -> str | None:
        """Generate top blocks summary when no critical alerts exist."""
        top_error_blocks = sorted(
            [
                (name, stats)
                for name, stats in block_stats.items()
                if stats.total_executions >= 10 and stats.failed_executions > 0
            ],
            key=lambda x: x[1].failed_executions,
            reverse=True,
        )[: self.include_top_blocks]

        if not top_error_blocks:
            return "✅ No errors reported for today - all blocks are running smoothly!"

        # Get error samples for top blocks
        for block_name, stats in top_error_blocks:
            if not stats.error_samples:
                stats.error_samples = self._get_error_samples_for_block(
                    stats.block_id, start_time, end_time, limit=2
                )

        count_text = (
            f"top {self.include_top_blocks}" if self.include_top_blocks > 1 else "top"
        )
        alert_msg = f"📊 Daily Error Summary - {count_text} blocks with most errors:"
        for block_name, stats in top_error_blocks:
            alert_msg += f"\n• {block_name}: {stats.failed_executions} errors ({stats.error_rate:.1f}% of {stats.total_executions})"

            if stats.error_samples:
                error_groups = self._group_similar_errors(stats.error_samples)
                if error_groups:
                    # Show most common error
                    most_common_error = next(iter(error_groups.items()))
                    alert_msg += f"\n  └ Most common: {most_common_error[0]}"

        return alert_msg

    def _get_error_samples_for_block(
        self, block_id: str, start_time: datetime, end_time: datetime, limit: int = 3
    ) -> list[str]:
        """Get error samples for a specific block - just a few recent ones."""
        # Only fetch a small number of recent failed executions for this specific block
        executions = execution_utils.get_db_client().get_node_executions(
            block_ids=[block_id],
            statuses=[ExecutionStatus.FAILED],
            created_time_gte=start_time,
            created_time_lte=end_time,
            limit=limit,  # Just get the limit we need
        )

        error_samples = []
        for execution in executions:
            if error_message := self._extract_error_message(execution):
                masked_error = self._mask_sensitive_data(error_message)
                error_samples.append(masked_error)

            if len(error_samples) >= limit:  # Stop once we have enough samples
                break

        return error_samples

    def _extract_error_message(self, execution: NodeExecutionResult) -> str | None:
        """Extract error message from execution output."""
        try:
            if execution.output_data and (
                error_msg := execution.output_data.get("error")
            ):
                return str(error_msg[0])
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


def report_block_error_rates(include_top_blocks: int | None = None):
    """Check block error rates and send Discord alerts if thresholds are exceeded."""
    monitor = BlockErrorMonitor(include_top_blocks=include_top_blocks)
    return monitor.check_block_error_rates()
