"""Unit tests for payment-lapse pause/resume of user automations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import NotificationType, PresetDeactivationReason

from backend.data.automation_pause import (
    SCHEDULE_PAUSE_REASON_PAYMENT_LAPSED,
    has_payment_lapsed_automations,
    pause_automations_for_payment_lapse,
    resume_automations_after_payment_restored,
)

_MODULE = "backend.data.automation_pause"
_PERSONAL_ORG = "personal-org-1"


def _mock_scheduler_client(paused: int = 0, resumed: int = 0) -> MagicMock:
    client = MagicMock()
    client.pause_user_graph_schedules = AsyncMock(return_value=paused)
    client.resume_user_graph_schedules = AsyncMock(return_value=resumed)
    return client


def _patches(client, preset_prisma, notify):
    return (
        patch(f"{_MODULE}.get_scheduler_client", return_value=client),
        patch("prisma.models.AgentPreset.prisma", preset_prisma),
        patch(f"{_MODULE}.queue_notification_async", new=notify),
        patch(
            f"{_MODULE}._get_personal_org_id",
            new=AsyncMock(return_value=_PERSONAL_ORG),
        ),
    )


@pytest.mark.asyncio
async def test_pause_marks_schedules_and_triggers_with_reason():
    client = _mock_scheduler_client(paused=2)
    preset_prisma = MagicMock()
    preset_prisma.return_value.update_many = AsyncMock(return_value=3)
    notify = AsyncMock()
    p1, p2, p3, p4 = _patches(client, preset_prisma, notify)
    with p1, p2, p3, p4:
        summary = await pause_automations_for_payment_lapse("user-1")

    assert summary.schedules == 2
    assert summary.triggers == 3
    client.pause_user_graph_schedules.assert_awaited_once_with(
        user_id="user-1",
        reason=SCHEDULE_PAUSE_REASON_PAYMENT_LAPSED,
        personal_org_id=_PERSONAL_ORG,
    )
    update_call = preset_prisma.return_value.update_many.call_args
    assert update_call.kwargs["where"] == {
        "userId": "user-1",
        "isActive": True,
        "isDeleted": False,
        "deactivationReason": None,
        "OR": [
            {"organizationId": None},
            {"organizationId": _PERSONAL_ORG},
        ],
    }
    assert update_call.kwargs["data"] == {
        "isActive": False,
        "deactivationReason": PresetDeactivationReason.PAYMENT_LAPSED,
    }
    event = notify.await_args.args[0]
    assert event.type == NotificationType.AUTOMATIONS_PAUSED
    assert event.data.paused_schedules == 2
    assert event.data.paused_triggers == 3


@pytest.mark.asyncio
async def test_pause_covers_personal_org_tagged_presets():
    """Presets carry the user's personal org id since org dual-write; the
    where-clause must match those, not only untagged legacy rows."""
    client = _mock_scheduler_client()
    preset_prisma = MagicMock()
    preset_prisma.return_value.update_many = AsyncMock(return_value=0)
    p1, p2, p3, p4 = _patches(client, preset_prisma, AsyncMock())
    with p1, p2, p3, p4:
        await pause_automations_for_payment_lapse("user-1")

    where = preset_prisma.return_value.update_many.call_args.kwargs["where"]
    assert {"organizationId": _PERSONAL_ORG} in where["OR"]
    assert where["deactivationReason"] is None
    assert where["isActive"] is True


@pytest.mark.asyncio
async def test_pause_falls_back_to_user_scope_without_personal_org():
    """If the personal org can't be resolved, pause everything the user owns
    (userId-only) rather than silently matching nothing — every preset is
    org-tagged after dual-write, so the org predicate would exclude all."""
    client = _mock_scheduler_client(paused=0)
    preset_prisma = MagicMock()
    preset_prisma.return_value.update_many = AsyncMock(return_value=0)
    with (
        patch(f"{_MODULE}.get_scheduler_client", return_value=client),
        patch("prisma.models.AgentPreset.prisma", preset_prisma),
        patch(f"{_MODULE}.queue_notification_async", new=AsyncMock()),
        patch(f"{_MODULE}._get_personal_org_id", new=AsyncMock(return_value=None)),
    ):
        await pause_automations_for_payment_lapse("user-1")

    where = preset_prisma.return_value.update_many.call_args.kwargs["where"]
    # Any org (untagged OR any organizationId) == userId-only scope.
    assert where["OR"] == [
        {"organizationId": None},
        {"organizationId": {"not": None}},
    ]
    assert where["userId"] == "user-1"
    client.pause_user_graph_schedules.assert_awaited_once_with(
        user_id="user-1",
        reason=SCHEDULE_PAUSE_REASON_PAYMENT_LAPSED,
        personal_org_id=None,
    )


@pytest.mark.asyncio
async def test_pause_with_nothing_to_pause_sends_no_notification():
    client = _mock_scheduler_client(paused=0)
    preset_prisma = MagicMock()
    preset_prisma.return_value.update_many = AsyncMock(return_value=0)
    notify = AsyncMock()
    p1, p2, p3, p4 = _patches(client, preset_prisma, notify)
    with p1, p2, p3, p4:
        summary = await pause_automations_for_payment_lapse("user-1")

    assert summary.total == 0
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_only_touches_payment_lapsed_automations():
    client = _mock_scheduler_client(resumed=1)
    preset_prisma = MagicMock()
    preset_prisma.return_value.update_many = AsyncMock(return_value=2)
    notify = AsyncMock()
    p1, p2, p3, p4 = _patches(client, preset_prisma, notify)
    with p1, p2, p3, p4:
        summary = await resume_automations_after_payment_restored("user-1")

    assert summary.schedules == 1
    assert summary.triggers == 2
    client.resume_user_graph_schedules.assert_awaited_once_with(
        user_id="user-1",
        reason=SCHEDULE_PAUSE_REASON_PAYMENT_LAPSED,
        personal_org_id=_PERSONAL_ORG,
    )
    update_call = preset_prisma.return_value.update_many.call_args
    assert update_call.kwargs["where"] == {
        "userId": "user-1",
        "isActive": False,
        "isDeleted": False,
        "deactivationReason": PresetDeactivationReason.PAYMENT_LAPSED,
        "OR": [
            {"organizationId": None},
            {"organizationId": _PERSONAL_ORG},
        ],
    }
    assert update_call.kwargs["data"] == {
        "isActive": True,
        "deactivationReason": None,
    }
    event = notify.await_args.args[0]
    assert event.type == NotificationType.AUTOMATIONS_RESUMED
    assert event.data.resumed_schedules == 1
    assert event.data.resumed_triggers == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("count,expected", [(2, True), (0, False)])
async def test_has_payment_lapsed_automations(count, expected):
    preset_prisma = MagicMock()
    preset_prisma.return_value.count = AsyncMock(return_value=count)
    with patch("prisma.models.AgentPreset.prisma", preset_prisma):
        assert await has_payment_lapsed_automations("user-1") is expected
    where = preset_prisma.return_value.count.call_args.kwargs["where"]
    assert where["deactivationReason"] == PresetDeactivationReason.PAYMENT_LAPSED
    assert where["userId"] == "user-1"


@pytest.mark.asyncio
async def test_resume_excludes_team_owned_presets():
    """A preset that became team-owned after being payment-lapsed must not be
    reactivated by the member's restored personal subscription — resume applies
    the same personal-org predicate as pause."""
    client = _mock_scheduler_client(resumed=0)
    preset_prisma = MagicMock()
    preset_prisma.return_value.update_many = AsyncMock(return_value=0)
    p1, p2, p3, p4 = _patches(client, preset_prisma, AsyncMock())
    with p1, p2, p3, p4:
        await resume_automations_after_payment_restored("user-1")

    where = preset_prisma.return_value.update_many.call_args.kwargs["where"]
    assert where["OR"] == [
        {"organizationId": None},
        {"organizationId": _PERSONAL_ORG},
    ]


@pytest.mark.asyncio
async def test_resume_with_nothing_to_resume_sends_no_notification():
    client = _mock_scheduler_client(resumed=0)
    preset_prisma = MagicMock()
    preset_prisma.return_value.update_many = AsyncMock(return_value=0)
    notify = AsyncMock()
    p1, p2, p3, p4 = _patches(client, preset_prisma, notify)
    with p1, p2, p3, p4:
        summary = await resume_automations_after_payment_restored("user-1")

    assert summary.total == 0
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_pause_deactivates_presets_even_when_scheduler_fails():
    client = MagicMock()
    client.pause_user_graph_schedules = AsyncMock(side_effect=RuntimeError("down"))
    preset_prisma = MagicMock()
    preset_prisma.return_value.update_many = AsyncMock(return_value=2)
    notify = AsyncMock()
    p1, p2, p3, p4 = _patches(client, preset_prisma, notify)
    with p1, p2, p3, p4:
        with pytest.raises(RuntimeError):
            await pause_automations_for_payment_lapse("user-1")

    preset_prisma.return_value.update_many.assert_awaited_once()
    event = notify.await_args.args[0]
    assert event.data.paused_triggers == 2


@pytest.mark.asyncio
async def test_resume_reactivates_presets_even_when_scheduler_fails():
    client = MagicMock()
    client.resume_user_graph_schedules = AsyncMock(side_effect=RuntimeError("down"))
    preset_prisma = MagicMock()
    preset_prisma.return_value.update_many = AsyncMock(return_value=1)
    notify = AsyncMock()
    p1, p2, p3, p4 = _patches(client, preset_prisma, notify)
    with p1, p2, p3, p4:
        with pytest.raises(RuntimeError):
            await resume_automations_after_payment_restored("user-1")

    preset_prisma.return_value.update_many.assert_awaited_once()
    event = notify.await_args.args[0]
    assert event.data.resumed_triggers == 1
