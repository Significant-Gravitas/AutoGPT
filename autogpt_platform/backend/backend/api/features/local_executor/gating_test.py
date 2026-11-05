from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.local_executor.gating import (
    is_local_executor_enabled,
    is_workflow_recording_enabled,
    local_executor_turned_off,
)
from backend.util.feature_flag import Flag


@pytest.mark.asyncio
async def test_deployment_kill_switch_prevents_flag_evaluation() -> None:
    get_flag = AsyncMock(return_value=True)

    with (
        patch(
            "backend.api.features.local_executor.gating.ChatConfig",
            return_value=MagicMock(use_local_pc_executor=False),
        ),
        patch(
            "backend.api.features.local_executor.gating.is_feature_enabled",
            get_flag,
        ),
    ):
        assert await is_local_executor_enabled("user-1") is False

    get_flag.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_executor_requires_per_user_flag() -> None:
    get_flag = AsyncMock(return_value=False)

    with (
        patch(
            "backend.api.features.local_executor.gating.ChatConfig",
            return_value=MagicMock(use_local_pc_executor=True),
        ),
        patch(
            "backend.api.features.local_executor.gating.is_feature_enabled",
            get_flag,
        ),
    ):
        assert await is_local_executor_enabled("user-1") is False

    get_flag.assert_awaited_once_with(Flag.LOCAL_PC_EXECUTOR, "user-1", default=False)


@pytest.mark.asyncio
async def test_recording_requires_both_feature_flags() -> None:
    get_flag = AsyncMock(
        side_effect=lambda flag, _user_id, default=False: flag
        in {Flag.LOCAL_PC_EXECUTOR, Flag.WORKFLOW_RECORDING}
    )

    with (
        patch(
            "backend.api.features.local_executor.gating.ChatConfig",
            return_value=MagicMock(use_local_pc_executor=True),
        ),
        patch(
            "backend.api.features.local_executor.gating.is_feature_enabled",
            get_flag,
        ),
    ):
        assert await is_workflow_recording_enabled("user-1") is True

    assert get_flag.await_args_list[0].args[:2] == (Flag.LOCAL_PC_EXECUTOR, "user-1")
    assert get_flag.await_args_list[1].args[:2] == (Flag.WORKFLOW_RECORDING, "user-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment", "flag_read", "turned_off"),
    [
        (False, (True, True), True),
        (True, (False, True), True),
        (True, (True, True), False),
        # A flag read that failed returns its default, which says nothing.
        (True, (False, False), False),
    ],
    ids=["deployment-off", "flag-off", "on", "flag-unreadable"],
)
async def test_an_open_connection_ends_only_on_a_real_off(
    deployment: bool, flag_read: tuple[bool, bool], turned_off: bool
) -> None:
    """A flag outage must not drop every connected machine at once."""
    with (
        patch(
            "backend.api.features.local_executor.gating.ChatConfig",
            return_value=MagicMock(use_local_pc_executor=deployment),
        ),
        patch(
            "backend.api.features.local_executor.gating.evaluate_feature_flag",
            AsyncMock(return_value=flag_read),
        ),
    ):
        assert await local_executor_turned_off("user-1") is turned_off
