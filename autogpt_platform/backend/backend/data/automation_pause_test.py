"""Unit tests for payment-lapse pause/resume of user automations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import NotificationType, PresetDeactivationReason

from backend.data.automation_pause import (
    SCHEDULE_PAUSE_REASON_PAYMENT_LAPSED,
    pause_automations_for_payment_lapse,
    resume_automations_after_payment_restored,
)

_MODULE = "backend.data.automation_pause"


def _mock_scheduler_client(paused: int = 0, resumed: int = 0) -> MagicMock:
    client = MagicMock()
    client.pause_user_graph_schedules = AsyncMock(return_value=paused)
    client.resume_user_graph_schedules = AsyncMock(return_value=resumed)
    return client


@pytest.mark.asyncio
async def test_pause_marks_schedules_and_triggers_with_reason():
    client = _mock_scheduler_client(paused=2)
    with (
        patch(f"{_MODULE}.get_scheduler_client", return_value=client),
        patch("prisma.models.AgentPreset.prisma") as mock_prisma,
        patch(f"{_MODULE}.queue_notification_async", new=AsyncMock()) as mock_notify,
    ):
        mock_prisma.return_value.update_many = AsyncMock(return_value=3)

        summary = await pause_automations_for_payment_lapse("user-1")

    assert summary.schedules == 2
    assert summary.triggers == 3
    client.pause_user_graph_schedules.assert_awaited_once_with(
        user_id="user-1", reason=SCHEDULE_PAUSE_REASON_PAYMENT_LAPSED
    )
    update_call = mock_prisma.return_value.update_many.call_args
    assert update_call.kwargs["where"] == {
        "userId": "user-1",
        "isActive": True,
        "isDeleted": False,
        "organizationId": None,
        "deactivationReason": None,
    }
    assert update_call.kwargs["data"] == {
        "isActive": False,
        "deactivationReason": PresetDeactivationReason.PAYMENT_LAPSED,
    }
    event = mock_notify.await_args.args[0]
    assert event.type == NotificationType.AUTOMATIONS_PAUSED
    assert event.data.paused_schedules == 2
    assert event.data.paused_triggers == 3


@pytest.mark.asyncio
async def test_pause_skips_user_deactivated_presets_via_where_clause():
    """The where clause must require deactivationReason=None so presets the
    user deactivated themselves are never stamped with PAYMENT_LAPSED."""
    client = _mock_scheduler_client()
    with (
        patch(f"{_MODULE}.get_scheduler_client", return_value=client),
        patch("prisma.models.AgentPreset.prisma") as mock_prisma,
        patch(f"{_MODULE}.queue_notification_async", new=AsyncMock()),
    ):
        mock_prisma.return_value.update_many = AsyncMock(return_value=0)
        await pause_automations_for_payment_lapse("user-1")

    where = mock_prisma.return_value.update_many.call_args.kwargs["where"]
    assert where["deactivationReason"] is None
    assert where["isActive"] is True


@pytest.mark.asyncio
async def test_pause_with_nothing_to_pause_sends_no_notification():
    client = _mock_scheduler_client(paused=0)
    with (
        patch(f"{_MODULE}.get_scheduler_client", return_value=client),
        patch("prisma.models.AgentPreset.prisma") as mock_prisma,
        patch(f"{_MODULE}.queue_notification_async", new=AsyncMock()) as mock_notify,
    ):
        mock_prisma.return_value.update_many = AsyncMock(return_value=0)

        summary = await pause_automations_for_payment_lapse("user-1")

    assert summary.total == 0
    mock_notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_only_touches_payment_lapsed_automations():
    client = _mock_scheduler_client(resumed=1)
    with (
        patch(f"{_MODULE}.get_scheduler_client", return_value=client),
        patch("prisma.models.AgentPreset.prisma") as mock_prisma,
        patch(f"{_MODULE}.queue_notification_async", new=AsyncMock()) as mock_notify,
    ):
        mock_prisma.return_value.update_many = AsyncMock(return_value=2)

        summary = await resume_automations_after_payment_restored("user-1")

    assert summary.schedules == 1
    assert summary.triggers == 2
    client.resume_user_graph_schedules.assert_awaited_once_with(
        user_id="user-1", reason=SCHEDULE_PAUSE_REASON_PAYMENT_LAPSED
    )
    update_call = mock_prisma.return_value.update_many.call_args
    assert update_call.kwargs["where"] == {
        "userId": "user-1",
        "isActive": False,
        "isDeleted": False,
        "deactivationReason": PresetDeactivationReason.PAYMENT_LAPSED,
    }
    assert update_call.kwargs["data"] == {
        "isActive": True,
        "deactivationReason": None,
    }
    event = mock_notify.await_args.args[0]
    assert event.type == NotificationType.AUTOMATIONS_RESUMED
    assert event.data.resumed_schedules == 1
    assert event.data.resumed_triggers == 2


@pytest.mark.asyncio
async def test_resume_with_nothing_to_resume_sends_no_notification():
    client = _mock_scheduler_client(resumed=0)
    with (
        patch(f"{_MODULE}.get_scheduler_client", return_value=client),
        patch("prisma.models.AgentPreset.prisma") as mock_prisma,
        patch(f"{_MODULE}.queue_notification_async", new=AsyncMock()) as mock_notify,
    ):
        mock_prisma.return_value.update_many = AsyncMock(return_value=0)

        summary = await resume_automations_after_payment_restored("user-1")

    assert summary.total == 0
    mock_notify.assert_not_awaited()
