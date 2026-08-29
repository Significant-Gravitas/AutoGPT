"""Unit tests for diagnostics data layer functions."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.diagnostics import (
    _calculate_total_runs,
    _detect_orphaned_schedules,
    get_execution_diagnostics,
    get_rabbitmq_cancel_queue_depth,
    get_rabbitmq_queue_depth,
)

# ---------------------------------------------------------------------------
# get_execution_diagnostics tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_execution_diagnostics_full():
    """Test get_execution_diagnostics aggregates all data correctly."""
    mock_row = {
        "running_count": 10,
        "queued_db_count": 5,
        "orphaned_running": 2,
        "orphaned_queued": 1,
        "failed_count_1h": 3,
        "failed_count_24h": 12,
        "stuck_running_24h": 1,
        "stuck_running_1h": 2,
        "stuck_queued_1h": 4,
        "queued_never_started": 3,
        "invalid_queued_with_start": 1,
        "invalid_running_without_start": 0,
        "completed_1h": 50,
        "completed_24h": 600,
    }

    mock_exec = MagicMock()
    mock_exec.started_at = datetime.now(timezone.utc) - timedelta(hours=48)

    with (
        patch(
            "backend.data.diagnostics.query_raw_with_schema",
            new_callable=AsyncMock,
            return_value=[mock_row],
        ),
        patch(
            "backend.data.diagnostics.get_rabbitmq_queue_depth",
            return_value=7,
        ),
        patch(
            "backend.data.diagnostics.get_rabbitmq_cancel_queue_depth",
            return_value=2,
        ),
        patch(
            "backend.data.diagnostics.get_graph_executions",
            new_callable=AsyncMock,
            return_value=[mock_exec],
        ),
    ):
        result = await get_execution_diagnostics()

    assert result.running_count == 10
    assert result.queued_db_count == 5
    assert result.orphaned_running == 2
    assert result.orphaned_queued == 1
    assert result.failed_count_1h == 3
    assert result.failed_count_24h == 12
    assert result.failure_rate_24h == 12 / 24.0
    assert result.stuck_running_24h == 1
    assert result.stuck_running_1h == 2
    assert result.stuck_queued_1h == 4
    assert result.queued_never_started == 3
    assert result.invalid_queued_with_start == 1
    assert result.invalid_running_without_start == 0
    assert result.completed_1h == 50
    assert result.completed_24h == 600
    assert result.throughput_per_hour == 600 / 24.0
    assert result.rabbitmq_queue_depth == 7
    assert result.cancel_queue_depth == 2
    assert result.oldest_running_hours is not None
    assert result.oldest_running_hours > 47.0


@pytest.mark.asyncio
async def test_get_execution_diagnostics_empty_db():
    """Test get_execution_diagnostics with empty database."""
    with (
        patch(
            "backend.data.diagnostics.query_raw_with_schema",
            new_callable=AsyncMock,
            return_value=[{}],
        ),
        patch(
            "backend.data.diagnostics.get_rabbitmq_queue_depth",
            return_value=-1,
        ),
        patch(
            "backend.data.diagnostics.get_rabbitmq_cancel_queue_depth",
            return_value=-1,
        ),
        patch(
            "backend.data.diagnostics.get_graph_executions",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = await get_execution_diagnostics()

    assert result.running_count == 0
    assert result.queued_db_count == 0
    assert result.failure_rate_24h == 0.0
    assert result.throughput_per_hour == 0.0
    assert result.oldest_running_hours is None
    assert result.rabbitmq_queue_depth == -1
    assert result.cancel_queue_depth == -1


@pytest.mark.asyncio
async def test_get_execution_diagnostics_no_started_at():
    """Test oldest_running_hours when oldest execution has no started_at."""
    mock_row = {
        "running_count": 1,
        "queued_db_count": 0,
        "orphaned_running": 0,
        "orphaned_queued": 0,
        "failed_count_1h": 0,
        "failed_count_24h": 0,
        "stuck_running_24h": 0,
        "stuck_running_1h": 0,
        "stuck_queued_1h": 0,
        "queued_never_started": 0,
        "invalid_queued_with_start": 0,
        "invalid_running_without_start": 1,
        "completed_1h": 0,
        "completed_24h": 0,
    }

    mock_exec = MagicMock()
    mock_exec.started_at = None

    with (
        patch(
            "backend.data.diagnostics.query_raw_with_schema",
            new_callable=AsyncMock,
            return_value=[mock_row],
        ),
        patch(
            "backend.data.diagnostics.get_rabbitmq_queue_depth",
            return_value=0,
        ),
        patch(
            "backend.data.diagnostics.get_rabbitmq_cancel_queue_depth",
            return_value=0,
        ),
        patch(
            "backend.data.diagnostics.get_graph_executions",
            new_callable=AsyncMock,
            return_value=[mock_exec],
        ),
    ):
        result = await get_execution_diagnostics()

    assert result.oldest_running_hours is None


# ---------------------------------------------------------------------------
# RabbitMQ queue depth tests
# ---------------------------------------------------------------------------


def test_rabbitmq_queue_depth_success():
    """Test successful RabbitMQ queue depth retrieval."""
    mock_method_frame = MagicMock()
    mock_method_frame.method.message_count = 42

    mock_channel = MagicMock()
    mock_channel.queue_declare.return_value = mock_method_frame

    mock_rabbitmq = MagicMock()
    mock_rabbitmq._channel = mock_channel

    with (
        patch(
            "backend.data.diagnostics.create_execution_queue_config",
            return_value=MagicMock(),
        ),
        patch(
            "backend.data.diagnostics.SyncRabbitMQ",
            return_value=mock_rabbitmq,
        ),
    ):
        result = get_rabbitmq_queue_depth()

    assert result == 42
    mock_rabbitmq.connect.assert_called_once()
    mock_rabbitmq.disconnect.assert_called_once()


def test_rabbitmq_queue_depth_connection_error():
    """Test RabbitMQ queue depth returns -1 on connection error."""
    mock_rabbitmq = MagicMock()
    mock_rabbitmq.connect.side_effect = Exception("Connection refused")

    with (
        patch(
            "backend.data.diagnostics.create_execution_queue_config",
            return_value=MagicMock(),
        ),
        patch(
            "backend.data.diagnostics.SyncRabbitMQ",
            return_value=mock_rabbitmq,
        ),
    ):
        result = get_rabbitmq_queue_depth()

    assert result == -1


def test_rabbitmq_queue_depth_no_channel():
    """Test RabbitMQ queue depth when channel is None."""
    mock_rabbitmq = MagicMock()
    mock_rabbitmq._channel = None

    with (
        patch(
            "backend.data.diagnostics.create_execution_queue_config",
            return_value=MagicMock(),
        ),
        patch(
            "backend.data.diagnostics.SyncRabbitMQ",
            return_value=mock_rabbitmq,
        ),
    ):
        result = get_rabbitmq_queue_depth()

    # Should return -1 because RuntimeError is caught
    assert result == -1


def test_rabbitmq_cancel_queue_depth_success():
    """Test successful RabbitMQ cancel queue depth retrieval."""
    mock_method_frame = MagicMock()
    mock_method_frame.method.message_count = 5

    mock_channel = MagicMock()
    mock_channel.queue_declare.return_value = mock_method_frame

    mock_rabbitmq = MagicMock()
    mock_rabbitmq._channel = mock_channel

    with (
        patch(
            "backend.data.diagnostics.create_execution_queue_config",
            return_value=MagicMock(),
        ),
        patch(
            "backend.data.diagnostics.SyncRabbitMQ",
            return_value=mock_rabbitmq,
        ),
    ):
        result = get_rabbitmq_cancel_queue_depth()

    assert result == 5


def test_rabbitmq_cancel_queue_depth_error():
    """Test RabbitMQ cancel queue depth returns -1 on error."""
    mock_rabbitmq = MagicMock()
    mock_rabbitmq.connect.side_effect = Exception("Connection refused")

    with (
        patch(
            "backend.data.diagnostics.create_execution_queue_config",
            return_value=MagicMock(),
        ),
        patch(
            "backend.data.diagnostics.SyncRabbitMQ",
            return_value=mock_rabbitmq,
        ),
    ):
        result = get_rabbitmq_cancel_queue_depth()

    assert result == -1


def test_rabbitmq_disconnect_error_handled():
    """Test that disconnect errors are handled gracefully."""
    mock_method_frame = MagicMock()
    mock_method_frame.method.message_count = 10

    mock_channel = MagicMock()
    mock_channel.queue_declare.return_value = mock_method_frame

    mock_rabbitmq = MagicMock()
    mock_rabbitmq._channel = mock_channel
    mock_rabbitmq.disconnect.side_effect = Exception("Disconnect failed")

    with (
        patch(
            "backend.data.diagnostics.create_execution_queue_config",
            return_value=MagicMock(),
        ),
        patch(
            "backend.data.diagnostics.SyncRabbitMQ",
            return_value=mock_rabbitmq,
        ),
    ):
        # Should still return the count even if disconnect fails
        result = get_rabbitmq_queue_depth()

    assert result == 10


# ---------------------------------------------------------------------------
# _calculate_total_runs tests
# ---------------------------------------------------------------------------


def test_calculate_total_runs_basic():
    """Test calculating total runs with a simple cron (every hour)."""
    now = datetime(2026, 4, 17, 0, 0, 0, tzinfo=timezone.utc)
    end = now + timedelta(hours=3)

    schedule = MagicMock()
    schedule.cron = "0 * * * *"  # Every hour

    result = _calculate_total_runs([schedule], now, end)
    assert result == 3  # 01:00, 02:00, 03:00


def test_calculate_total_runs_invalid_cron():
    """Test that invalid cron expressions are skipped."""
    now = datetime(2026, 4, 17, 0, 0, 0, tzinfo=timezone.utc)
    end = now + timedelta(hours=1)

    schedule = MagicMock()
    schedule.cron = "invalid cron expression"

    result = _calculate_total_runs([schedule], now, end)
    assert result == 0


def test_calculate_total_runs_multiple_schedules():
    """Test total runs across multiple schedules."""
    now = datetime(2026, 4, 17, 0, 0, 0, tzinfo=timezone.utc)
    end = now + timedelta(hours=2)

    sched1 = MagicMock()
    sched1.cron = "0 * * * *"  # Every hour

    sched2 = MagicMock()
    sched2.cron = "*/30 * * * *"  # Every 30 min

    result = _calculate_total_runs([sched1, sched2], now, end)
    # sched1: 01:00, 02:00 = 2
    # sched2: 00:30, 01:00, 01:30, 02:00 = 4
    assert result == 6


def test_calculate_total_runs_empty():
    """Test with no schedules."""
    now = datetime(2026, 4, 17, 0, 0, 0, tzinfo=timezone.utc)
    end = now + timedelta(hours=1)

    result = _calculate_total_runs([], now, end)
    assert result == 0


# ---------------------------------------------------------------------------
# _detect_orphaned_schedules tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_orphaned_schedules_deleted_graph():
    """Test detection of schedules with deleted graphs."""
    schedule = MagicMock()
    schedule.id = "sched-1"
    schedule.graph_id = "graph-deleted"
    schedule.graph_version = 1
    schedule.user_id = "user-1"

    with patch("backend.data.diagnostics.AgentGraph.prisma") as mock_graph_prisma:
        mock_graph_prisma.return_value.find_unique = AsyncMock(return_value=None)

        result = await _detect_orphaned_schedules([schedule])

    assert "sched-1" in result["deleted_graph"]
    assert len(result["no_library_access"]) == 0


@pytest.mark.asyncio
async def test_detect_orphaned_schedules_no_library_access():
    """Test detection of schedules where user lost library access."""
    schedule = MagicMock()
    schedule.id = "sched-2"
    schedule.graph_id = "graph-1"
    schedule.graph_version = 1
    schedule.user_id = "user-2"

    mock_graph = MagicMock()

    with (
        patch("backend.data.diagnostics.AgentGraph.prisma") as mock_graph_prisma,
        patch("backend.data.diagnostics.LibraryAgent.prisma") as mock_lib_prisma,
    ):
        mock_graph_prisma.return_value.find_unique = AsyncMock(return_value=mock_graph)
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=None)

        result = await _detect_orphaned_schedules([schedule])

    assert "sched-2" in result["no_library_access"]
    assert len(result["deleted_graph"]) == 0


@pytest.mark.asyncio
async def test_detect_orphaned_schedules_validation_error():
    """Test detection of schedules that fail validation."""
    schedule = MagicMock()
    schedule.id = "sched-3"
    schedule.graph_id = "graph-1"
    schedule.graph_version = 1
    schedule.user_id = "user-3"

    with patch("backend.data.diagnostics.AgentGraph.prisma") as mock_graph_prisma:
        mock_graph_prisma.return_value.find_unique = AsyncMock(
            side_effect=Exception("DB connection error")
        )

        result = await _detect_orphaned_schedules([schedule])

    assert "sched-3" in result["validation_failed"]


@pytest.mark.asyncio
async def test_detect_orphaned_schedules_healthy():
    """Test that healthy schedules are not flagged."""
    schedule = MagicMock()
    schedule.id = "sched-ok"
    schedule.graph_id = "graph-1"
    schedule.graph_version = 1
    schedule.user_id = "user-1"

    mock_graph = MagicMock()
    mock_library_agent = MagicMock()

    with (
        patch("backend.data.diagnostics.AgentGraph.prisma") as mock_graph_prisma,
        patch("backend.data.diagnostics.LibraryAgent.prisma") as mock_lib_prisma,
    ):
        mock_graph_prisma.return_value.find_unique = AsyncMock(return_value=mock_graph)
        mock_lib_prisma.return_value.find_first = AsyncMock(
            return_value=mock_library_agent
        )

        result = await _detect_orphaned_schedules([schedule])

    assert len(result["deleted_graph"]) == 0
    assert len(result["no_library_access"]) == 0
    assert len(result["validation_failed"]) == 0
