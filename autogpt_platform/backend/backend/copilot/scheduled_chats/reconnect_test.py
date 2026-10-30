import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.scheduled_chats import reconnect
from backend.copilot.scheduled_chats.outcomes import classify_run, tool_failure_detail
from backend.copilot.tools.expert_scope import provider_slug
from backend.copilot.tools.helpers import _single_provider
from backend.copilot.tools.models import ErrorResponse
from backend.integrations.providers import ProviderName


@pytest.mark.asyncio
async def test_a_connect_does_not_wait_for_the_resume(mocker):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_resume(user_id: str, provider: str) -> None:
        started.set()
        await release.wait()

    mocker.patch.object(reconnect, "resume_after_reconnect", side_effect=slow_resume)

    task = reconnect.resume_after_reconnect_in_background("user-1", "discord")
    await started.wait()

    assert not task.done()
    release.set()
    await task


def _told_row() -> SimpleNamespace:
    """What the reconnect read: a chat only told about Discord."""
    return SimpleNamespace(id="row-1", scheduleKey="sched-1", pausedAt=None)


def _patch_rows(mocker, deleted: int) -> tuple[MagicMock, AsyncMock]:
    rows = MagicMock(delete_many=AsyncMock(return_value=deleted))
    mocker.patch.object(
        reconnect.ScheduledChatStreak, "prisma", return_value=rows, create=True
    )
    resume = mocker.patch.object(reconnect, "resume_scheduled_chat", AsyncMock())
    return rows, resume


@pytest.mark.asyncio
async def test_a_chat_only_told_is_forgotten_while_it_stays_unpaused(mocker):
    rows, resume = _patch_rows(mocker, deleted=1)

    await reconnect._clear("user-1", _told_row())

    rows.delete_many.assert_awaited_once_with(where={"id": "row-1", "pausedAt": None})
    resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_chat_paused_since_it_was_read_is_resumed_not_dropped(mocker):
    """A run still failing on the old connection paused it after the read.
    Dropping the row would leave its jobs paused with nothing to resume them."""
    _, resume = _patch_rows(mocker, deleted=0)

    await reconnect._clear("user-1", _told_row())

    resume.assert_awaited_once_with("user-1", "sched-1")


@pytest.mark.parametrize("provider", list(ProviderName))
def test_a_failing_tool_names_its_provider_as_the_connect_route_does(provider):
    """A reconnect resumes the chats whose streak names the provider just
    connected: the connect route's ProviderName value against what the
    failing tool put in ErrorResponse.details. One vocabulary, kept so."""
    output = ErrorResponse(
        message="The saved credential was refused.",
        error="credential_rejected",
        details=_single_provider({provider_slug(provider)}),
    ).model_dump_json()
    detail = tool_failure_detail(output)
    assert detail is not None

    outcome = classify_run(
        cancelled=False,
        turn_error_code=None,
        turn_errored=False,
        tool_failures=[("run_block", detail)],
        tool_successes=0,
    )

    assert outcome.failure is not None
    assert outcome.failure.kind == "connection"
    assert outcome.failure.provider == provider.value
