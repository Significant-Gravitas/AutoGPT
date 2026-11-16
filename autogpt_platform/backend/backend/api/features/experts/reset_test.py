"""The reset's computer erase against a mocked E2B — no DB, no E2B account."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.experts import reset

_MODULE = "backend.api.features.experts.reset"


def _config(api_key: str | None) -> MagicMock:
    return MagicMock(return_value=SimpleNamespace(active_e2b_api_key=api_key))


@pytest.mark.asyncio
async def test_erase_computer_kills_the_box_then_destroys_its_volume():
    volumes = [
        SimpleNamespace(volume_id="vol-user", name="autogpt-user-u1"),
        SimpleNamespace(volume_id="vol-expert", name="autogpt-expert-e1"),
        SimpleNamespace(volume_id="vol-other", name="autogpt-expert-e2"),
    ]
    calls: list[str] = []
    kill = AsyncMock(side_effect=lambda *a, **k: calls.append("kill") or True)
    volume = MagicMock()
    volume.list = AsyncMock(return_value=volumes)
    volume.destroy = AsyncMock(
        side_effect=lambda *a, **k: calls.append("destroy") or True
    )
    with (
        patch(f"{_MODULE}.ChatConfig", _config("e2b-key")),
        patch(f"{_MODULE}.kill_expert_sandbox", kill),
        patch(f"{_MODULE}.AsyncVolume", volume),
    ):
        assert await reset._erase_computer("e1") is True

    kill.assert_awaited_once_with("e1", "e2b-key")
    # Only the expert's own volume goes: the user's shared one and another
    # expert's stay.
    volume.destroy.assert_awaited_once_with("vol-expert", api_key="e2b-key")
    # The box is gone before its volume is, so nothing is writing to it.
    assert calls == ["kill", "destroy"]


@pytest.mark.asyncio
async def test_erase_computer_without_a_volume_reports_nothing_erased():
    volume = MagicMock()
    volume.list = AsyncMock(return_value=[])
    volume.destroy = AsyncMock()
    with (
        patch(f"{_MODULE}.ChatConfig", _config("e2b-key")),
        patch(f"{_MODULE}.kill_expert_sandbox", AsyncMock(return_value=False)),
        patch(f"{_MODULE}.find_owned_sandbox_id", AsyncMock(return_value=None)),
        patch(f"{_MODULE}.AsyncVolume", volume),
    ):
        assert await reset._erase_computer("e1") is False
    volume.destroy.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_box_that_survives_the_kill_fails_the_reset():
    """kill_expert_sandbox answers False both for "no box" and for a failed
    kill; a box still there would come back with its old state."""
    volume = MagicMock()
    volume.list = AsyncMock(return_value=[])
    with (
        patch(f"{_MODULE}.ChatConfig", _config("e2b-key")),
        patch(f"{_MODULE}.kill_expert_sandbox", AsyncMock(return_value=False)),
        patch(f"{_MODULE}.find_owned_sandbox_id", AsyncMock(return_value="sbx-1")),
        patch(f"{_MODULE}.AsyncVolume", volume),
        pytest.raises(reset.ExpertResetError),
    ):
        await reset._erase_computer("e1")
    volume.list.assert_not_called()


@pytest.mark.asyncio
async def test_erase_computer_without_e2b_touches_nothing():
    kill = AsyncMock()
    volume = MagicMock()
    with (
        patch(f"{_MODULE}.ChatConfig", _config(None)),
        patch(f"{_MODULE}.kill_expert_sandbox", kill),
        patch(f"{_MODULE}.AsyncVolume", volume),
    ):
        assert await reset._erase_computer("e1") is False
    kill.assert_not_awaited()
    volume.list.assert_not_called()


@pytest.mark.asyncio
async def test_a_failed_volume_delete_fails_the_reset():
    """The reset route answers 503 so the user can retry, rather than claiming
    an erase that did not happen."""
    volume = MagicMock()
    volume.list = AsyncMock(
        return_value=[SimpleNamespace(volume_id="v", name="autogpt-expert-e1")]
    )
    volume.destroy = AsyncMock(side_effect=RuntimeError("E2B is down"))
    with (
        patch(f"{_MODULE}.ChatConfig", _config("e2b-key")),
        patch(f"{_MODULE}.kill_expert_sandbox", AsyncMock(return_value=True)),
        patch(f"{_MODULE}.AsyncVolume", volume),
        pytest.raises(RuntimeError),
    ):
        await reset._erase_computer("e1")
