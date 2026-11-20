from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.local_executor.gating import (
    is_computer_use_enabled,
    is_local_executor_enabled,
    is_workflow_recording_enabled,
    local_executor_turned_off,
)
from backend.util.feature_flag import Flag


def _flags(*on: Flag) -> AsyncMock:
    return AsyncMock(side_effect=lambda flag, _user_id, default=False: flag in on)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_local_executor_is_its_per_user_flag_alone(enabled: bool) -> None:
    """One switch per feature: the rollout flag, with no environment setting
    beside it to keep in step across deployments."""
    get_flag = AsyncMock(return_value=enabled)

    with patch(
        "backend.api.features.local_executor.gating.is_feature_enabled", get_flag
    ):
        assert await is_local_executor_enabled("user-1") is enabled

    get_flag.assert_awaited_once_with(Flag.LOCAL_PC_EXECUTOR, "user-1", default=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("on", "enabled"),
    [
        ((Flag.LOCAL_PC_EXECUTOR, Flag.LOCAL_PC_COMPUTER_USE), True),
        ((Flag.LOCAL_PC_EXECUTOR,), False),
        # Computer use is a part of Local PC, never without it.
        ((Flag.LOCAL_PC_COMPUTER_USE,), False),
    ],
    ids=["both", "local-pc-only", "computer-use-only"],
)
async def test_computer_use_needs_its_own_flag_and_local_pcs(
    on: tuple[Flag, ...], enabled: bool
) -> None:
    with patch(
        "backend.api.features.local_executor.gating.is_feature_enabled", _flags(*on)
    ):
        assert await is_computer_use_enabled("user-1") is enabled


@pytest.mark.asyncio
async def test_recording_requires_both_feature_flags() -> None:
    get_flag = _flags(Flag.LOCAL_PC_EXECUTOR, Flag.WORKFLOW_RECORDING)

    with patch(
        "backend.api.features.local_executor.gating.is_feature_enabled", get_flag
    ):
        assert await is_workflow_recording_enabled("user-1") is True

    assert get_flag.await_args_list[0].args[:2] == (Flag.LOCAL_PC_EXECUTOR, "user-1")
    assert get_flag.await_args_list[1].args[:2] == (Flag.WORKFLOW_RECORDING, "user-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("flag_read", "turned_off"),
    [
        ((False, True), True),
        ((True, True), False),
        # A flag read that failed returns its default, which says nothing.
        ((False, False), False),
    ],
    ids=["flag-off", "on", "flag-unreadable"],
)
async def test_an_open_connection_ends_only_on_a_real_off(
    flag_read: tuple[bool, bool], turned_off: bool
) -> None:
    """A flag outage must not drop every connected machine at once."""
    with patch(
        "backend.api.features.local_executor.gating.evaluate_feature_flag",
        AsyncMock(return_value=flag_read),
    ):
        assert await local_executor_turned_off("user-1") is turned_off
