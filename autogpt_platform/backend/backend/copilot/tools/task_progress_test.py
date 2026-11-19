"""What a task's worker may write on its task, and how short it must be."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import turn_end
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.task_models import (
    TaskEvidenceAddedResponse,
    TaskFinishedResponse,
    TaskProgressResponse,
)
from backend.copilot.tools.task_progress import (
    AddEvidenceTool,
    FinishTaskTool,
    ReportProgressTool,
)
from backend.data.task_result import TaskFinish
from backend.data.task_text import (
    CLAIM_MAX_CHARS,
    PROGRESS_MAX_CHARS,
    SUMMARY_MAX_CHARS,
)

from ._test_data import make_session

_USER = "test-user-task-progress"


@pytest.fixture
def store():
    store = MagicMock()
    store.set_progress = AsyncMock(return_value=MagicMock())
    store.add_evidence = AsyncMock(return_value=MagicMock(id="ev-1", verified=True))
    store.finish_task = AsyncMock(
        return_value=TaskFinish(task_id="task-1", document_title="brief.md")
    )
    with (
        patch("backend.copilot.tools.task_progress.task_record_db", return_value=store),
        patch("backend.copilot.tools.task_progress.task_result_db", return_value=store),
    ):
        yield store


@pytest.fixture
def session():
    session = make_session(_USER)
    yield session
    turn_end.clear(session.session_id)


async def test_a_progress_line_is_noted_without_sending_the_task_back(store, session):
    result = await ReportProgressTool()._execute(
        _USER, session, status_text="Comparing pricing pages", next_step="Write it up"
    )
    assert isinstance(result, TaskProgressResponse)
    assert set(result.model_dump(exclude_none=True)) <= {
        "type",
        "message",
        "session_id",
    }
    store.set_progress.assert_awaited_once_with(
        session.session_id,
        _USER,
        "Comparing pricing pages",
        "Write it up",
        session.expert_id,
    )


@pytest.mark.parametrize("name", ["message", "text", "status"])
async def test_a_progress_line_under_another_argument_name_still_lands(
    store, session, name
):
    result = await ReportProgressTool()._execute(
        _USER, session, **{name: "Reading the pricing pages"}
    )
    assert isinstance(result, TaskProgressResponse)
    assert store.set_progress.await_args.args[2] == "Reading the pricing pages"


async def test_a_long_progress_line_is_refused_with_the_cap(store, session):
    result = await ReportProgressTool()._execute(
        _USER, session, status_text="Found that " + "x" * PROGRESS_MAX_CHARS
    )
    assert isinstance(result, ErrorResponse)
    assert str(PROGRESS_MAX_CHARS) in result.message
    store.set_progress.assert_not_awaited()


async def test_a_long_next_step_is_cut_rather_than_refused(store, session):
    await ReportProgressTool()._execute(
        _USER, session, status_text="Writing", next_step="word " * 60
    )
    next_step = store.set_progress.await_args.args[3]
    assert len(next_step) <= PROGRESS_MAX_CHARS


@pytest.mark.parametrize("name", ["claim", "description", "summary"])
async def test_proof_is_attached_whatever_the_claim_is_called(store, session, name):
    result = await AddEvidenceTool()._execute(
        _USER, session, **{name: "Reminder sent to Acme"}, tool="send_email"
    )
    assert isinstance(result, TaskEvidenceAddedResponse) and result.verified
    evidence = store.add_evidence.await_args.args[2]
    assert (evidence.claim, evidence.tool) == ("Reminder sent to Acme", "send_email")


async def test_a_long_claim_is_refused_with_the_cap(store, session):
    result = await AddEvidenceTool()._execute(
        _USER, session, claim="Sent " + "x" * CLAIM_MAX_CHARS
    )
    assert isinstance(result, ErrorResponse)
    assert str(CLAIM_MAX_CHARS) in result.message
    store.add_evidence.assert_not_awaited()


async def test_finishing_saves_the_summary_and_names_the_document(store, session):
    result = await FinishTaskTool()._execute(
        _USER, session, summary="Brief ready: Pro is $20 a month.", file="brief.md"
    )
    assert isinstance(result, TaskFinishedResponse)
    assert result.document == "brief.md"
    store.finish_task.assert_awaited_once_with(
        session.session_id, _USER, "Brief ready: Pro is $20 a month.", "brief.md"
    )


async def test_a_summary_over_the_cap_is_refused_and_pointed_at_the_file(
    store, session
):
    result = await FinishTaskTool()._execute(
        _USER, session, summary="word " * SUMMARY_MAX_CHARS
    )
    assert isinstance(result, ErrorResponse)
    assert str(SUMMARY_MAX_CHARS) in result.message and "file" in result.message
    store.finish_task.assert_not_awaited()


async def test_finishing_with_a_file_the_task_never_saved_is_refused(store, session):
    store.finish_task.return_value = TaskFinish(task_id="task-1", missing_file=True)
    result = await FinishTaskTool()._execute(
        _USER, session, summary="Done.", file="ghost.md"
    )
    assert isinstance(result, ErrorResponse)
    assert "ghost.md" in result.message and "write_workspace_file" in result.message


@pytest.mark.parametrize(
    "tool, args",
    [
        (ReportProgressTool(), {"status_text": "Reading"}),
        (AddEvidenceTool(), {"claim": "Sent"}),
        (FinishTaskTool(), {"summary": "Done."}),
    ],
)
async def test_a_chat_working_on_no_task_is_told_so(store, session, tool, args):
    store.set_progress.return_value = None
    store.add_evidence.return_value = None
    store.finish_task.return_value = None
    result = await tool._execute(_USER, session, **args)
    assert isinstance(result, ErrorResponse)
    assert "not working on a task" in result.message


async def test_a_background_worker_that_finishes_ends_its_turn(store, session):
    store.finish_task.return_value = TaskFinish(task_id="task-1", in_background=True)
    result = await FinishTaskTool()._execute(_USER, session, summary="Done.")

    assert isinstance(result, TaskFinishedResponse)
    # No further model call, and anything started later is refused.
    assert turn_end.is_ending(session.session_id)
    assert turn_end.refusal(session.session_id, "web_search") == (
        turn_end.FINISHED_REFUSAL
    )


async def test_finishing_in_a_chat_the_user_is_in_leaves_room_for_a_closing_line(
    store, session
):
    result = await FinishTaskTool()._execute(_USER, session, summary="Done.")

    assert isinstance(result, TaskFinishedResponse)
    assert not turn_end.is_ending(session.session_id)
    assert "one line" in result.message


async def test_a_refused_finish_does_not_end_the_turn(store, session):
    store.finish_task.return_value = TaskFinish(
        task_id="task-1", missing_file=True, in_background=True
    )
    await FinishTaskTool()._execute(_USER, session, summary="Done.", file="ghost.md")
    assert not turn_end.is_ending(session.session_id)
