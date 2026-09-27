"""The memory_forget_confirm tool when memory is busy: a forget that found
an ingestion holding the graph's write lock wrote nothing, so the tool waits
five seconds and tries once more, then reports it. ``retract`` is mocked;
its busy failure is pinned in ``graphiti/recall_forget_test.py``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.graphiti.memory_model import (
    ForgetResult,
    MemoryForgetFailure,
    MemoryForgetFailureCode,
)
from backend.copilot.model import ChatSession
from backend.copilot.tools.graphiti_forget import MemoryForgetConfirmTool
from backend.copilot.tools.models import MemoryForgetConfirmResponse

_MODULE = "backend.copilot.tools.graphiti_forget"


async def _enabled(_user_id: str) -> bool:
    return True


class TestForgetConfirmWhenMemoryIsBusy:
    """A forget that found memory busy wrote nothing: the tool waits five
    seconds and tries once more, then reports it."""

    @pytest.mark.asyncio
    async def test_a_busy_forget_is_tried_once_more(self) -> None:
        busy = ForgetResult(failures=[MemoryForgetFailure.busy("e1")])
        retract = AsyncMock(side_effect=[busy, ForgetResult(deleted=["e1"])])
        sleep = AsyncMock()
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.retract", retract),
            patch(f"{_MODULE}.asyncio.sleep", sleep),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["e1"]
            )

        assert isinstance(response, MemoryForgetConfirmResponse)
        assert (response.deleted_uuids, response.failed_uuids) == (["e1"], [])
        sleep.assert_awaited_once_with(5)
        assert retract.await_count == 2

    @pytest.mark.asyncio
    async def test_busy_twice_is_reported_as_busy(self) -> None:
        busy = ForgetResult(failures=[MemoryForgetFailure.busy("e1")])
        retract = AsyncMock(return_value=busy)
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.retract", retract),
            patch(f"{_MODULE}.asyncio.sleep", AsyncMock()),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["e1"]
            )

        assert isinstance(response, MemoryForgetConfirmResponse)
        assert [f.code for f in response.failures] == [MemoryForgetFailureCode.BUSY]
        assert "nothing was forgotten" in response.message
        assert retract.await_count == 2

    @pytest.mark.asyncio
    async def test_any_other_failure_is_not_retried(self) -> None:
        missing = ForgetResult(failures=[MemoryForgetFailure.no_match("e1")])
        retract = AsyncMock(return_value=missing)
        sleep = AsyncMock()
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.retract", retract),
            patch(f"{_MODULE}.asyncio.sleep", sleep),
        ):
            await MemoryForgetConfirmTool()._execute("user-abc", session, uuids=["e1"])

        sleep.assert_not_awaited()
        retract.assert_awaited_once()
