"""Tests for BlockStatsWithSamples and BlockErrorStats user API key error filtering."""

from unittest.mock import MagicMock, patch

import pytest

from backend.data.execution import BlockErrorStats
from backend.monitoring.block_error_monitor import (
    BlockErrorMonitor,
    BlockStatsWithSamples,
)


class TestBlockErrorStats:
    def test_error_rate_all_failed(self):
        stats = BlockErrorStats(
            block_id="b1", total_executions=10, failed_executions=10
        )
        assert stats.error_rate == 100.0

    def test_error_rate_none_failed(self):
        stats = BlockErrorStats(block_id="b1", total_executions=10, failed_executions=0)
        assert stats.error_rate == 0.0

    def test_error_rate_zero_total(self):
        stats = BlockErrorStats(block_id="b1", total_executions=0, failed_executions=0)
        assert stats.error_rate == 0.0

    def test_platform_failed_excludes_user_api_key_errors(self):
        stats = BlockErrorStats(
            block_id="b1",
            total_executions=100,
            failed_executions=50,
            user_api_key_error_executions=40,
        )
        assert stats.platform_failed_executions == 10

    def test_platform_error_rate_excludes_user_api_key_errors(self):
        stats = BlockErrorStats(
            block_id="b1",
            total_executions=100,
            failed_executions=100,
            user_api_key_error_executions=100,
        )
        assert stats.platform_error_rate == 0.0

    def test_platform_error_rate_partial_user_errors(self):
        stats = BlockErrorStats(
            block_id="b1",
            total_executions=200,
            failed_executions=100,
            user_api_key_error_executions=80,
        )
        assert stats.platform_error_rate == 10.0

    def test_platform_error_rate_no_user_errors(self):
        stats = BlockErrorStats(
            block_id="b1",
            total_executions=100,
            failed_executions=50,
        )
        assert stats.platform_error_rate == 50.0

    def test_user_api_key_error_executions_defaults_to_zero(self):
        stats = BlockErrorStats(block_id="b1", total_executions=10, failed_executions=5)
        assert stats.user_api_key_error_executions == 0


class TestBlockStatsWithSamples:
    def test_platform_failed_excludes_user_api_key_errors(self):
        stats = BlockStatsWithSamples(
            block_id="b1",
            block_name="TestBlock",
            total_executions=100,
            failed_executions=50,
            user_api_key_error_executions=40,
        )
        assert stats.platform_failed_executions == 10

    def test_platform_error_rate_all_user_errors(self):
        stats = BlockStatsWithSamples(
            block_id="b1",
            block_name="TestBlock",
            total_executions=240,
            failed_executions=240,
            user_api_key_error_executions=240,
        )
        assert stats.platform_error_rate == 0.0

    def test_platform_error_rate_zero_total(self):
        stats = BlockStatsWithSamples(
            block_id="b1",
            block_name="TestBlock",
            total_executions=0,
            failed_executions=0,
        )
        assert stats.platform_error_rate == 0.0

    def test_error_rate_uses_total_failures(self):
        stats = BlockStatsWithSamples(
            block_id="b1",
            block_name="TestBlock",
            total_executions=100,
            failed_executions=60,
            user_api_key_error_executions=40,
        )
        assert stats.error_rate == 60.0
        assert stats.platform_error_rate == 20.0


class TestGenerateCriticalAlerts:
    def _make_monitor(self, threshold: float = 0.5):
        mock_config = MagicMock()
        mock_config.block_error_rate_threshold = threshold
        mock_config.block_error_include_top_blocks = 5
        with patch("backend.monitoring.block_error_monitor.config", mock_config):
            with patch(
                "backend.monitoring.block_error_monitor.get_notification_manager_client"
            ):
                monitor = BlockErrorMonitor()
        monitor.config = mock_config
        return monitor

    def test_no_alert_when_platform_error_rate_below_threshold(self):
        monitor = self._make_monitor(threshold=0.5)
        stats = {
            "TestBlock": BlockStatsWithSamples(
                block_id="b1",
                block_name="TestBlock",
                total_executions=240,
                failed_executions=240,
                user_api_key_error_executions=240,
            )
        }
        alerts = monitor._generate_critical_alerts(stats, threshold=0.5)
        assert alerts == []

    def test_alert_when_platform_error_rate_exceeds_threshold(self):
        monitor = self._make_monitor(threshold=0.5)
        stats = {
            "TestBlock": BlockStatsWithSamples(
                block_id="b1",
                block_name="TestBlock",
                total_executions=100,
                failed_executions=80,
                user_api_key_error_executions=0,
            )
        }
        alerts = monitor._generate_critical_alerts(stats, threshold=0.5)
        assert len(alerts) == 1
        assert "80.0%" in alerts[0]
        assert "TestBlock" in alerts[0]

    def test_alert_includes_user_key_note_when_present(self):
        monitor = self._make_monitor(threshold=0.5)
        stats = {
            "TestBlock": BlockStatsWithSamples(
                block_id="b1",
                block_name="TestBlock",
                total_executions=100,
                failed_executions=90,
                user_api_key_error_executions=10,
            )
        }
        alerts = monitor._generate_critical_alerts(stats, threshold=0.5)
        assert len(alerts) == 1
        assert "user-supplied invalid API keys" in alerts[0]
        assert "10 additional failure" in alerts[0]

    def test_no_user_key_note_when_no_user_errors(self):
        monitor = self._make_monitor(threshold=0.5)
        stats = {
            "TestBlock": BlockStatsWithSamples(
                block_id="b1",
                block_name="TestBlock",
                total_executions=100,
                failed_executions=80,
                user_api_key_error_executions=0,
            )
        }
        alerts = monitor._generate_critical_alerts(stats, threshold=0.5)
        assert len(alerts) == 1
        assert "user-supplied" not in alerts[0]
