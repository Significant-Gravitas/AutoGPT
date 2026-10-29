from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.scheduled_chats import jobs
from backend.util.exceptions import NotFoundError

_USER = "user-1"


@pytest.fixture
def scheduler(mocker) -> MagicMock:
    client = MagicMock(
        pause_schedule=AsyncMock(return_value=True),
        resume_schedule=AsyncMock(return_value=True),
        get_execution_schedules=AsyncMock(return_value=[]),
    )
    mocker.patch.object(jobs, "get_scheduler_client", return_value=client)
    return client


def _fail_on(failing: str, error: Exception):
    async def act(schedule_id: str, *, user_id: str) -> bool:
        if schedule_id == failing:
            raise error
        return True

    return act


@pytest.mark.asyncio
async def test_every_job_is_paused(scheduler: MagicMock):
    assert await jobs.pause_jobs(_USER, ["a", "b"]) == ["a", "b"]


@pytest.mark.asyncio
async def test_a_job_its_owner_paused_is_not_ours_to_restart(scheduler):
    scheduler.pause_schedule.side_effect = lambda schedule_id, *, user_id: (
        schedule_id != "a"
    )

    assert await jobs.pause_jobs(_USER, ["a", "b"]) == ["b"]


@pytest.mark.asyncio
async def test_a_job_deleted_since_is_skipped_and_the_rest_pause(scheduler):
    scheduler.pause_schedule.side_effect = _fail_on("b", NotFoundError("gone"))

    assert await jobs.pause_jobs(_USER, ["a", "b"]) == ["a"]
    scheduler.resume_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_with_no_live_job_there_is_nothing_to_pause(scheduler):
    scheduler.pause_schedule.side_effect = NotFoundError("gone")

    assert await jobs.pause_jobs(_USER, ["a"]) == []
    assert await jobs.pause_jobs(_USER, []) == []


@pytest.mark.asyncio
async def test_a_pause_that_fails_resumes_only_what_it_paused(scheduler):
    async def pause(schedule_id: str, *, user_id: str) -> bool:
        if schedule_id == "c":
            raise RuntimeError("scheduler refused")
        # "b" was already paused by its owner, and stays so.
        return schedule_id != "b"

    scheduler.pause_schedule.side_effect = pause

    assert await jobs.pause_jobs(_USER, ["a", "b", "c"]) is None
    scheduler.resume_schedule.assert_awaited_once_with("a", user_id=_USER)


@pytest.mark.asyncio
async def test_a_pause_whose_answer_was_lost_is_put_back_too(scheduler):
    """The scheduler paused "b" and the answer never came back. Left out of
    the put-back, it would stay paused with no streak saying so, and nothing
    would ever resume it."""
    scheduler.get_execution_schedules.return_value = [
        MagicMock(id="a"),
        MagicMock(id="b"),
    ]
    scheduler.pause_schedule.side_effect = _fail_on("b", TimeoutError("no answer"))

    assert await jobs.pause_jobs(_USER, ["a", "b"]) is None
    resumed = [c.args[0] for c in scheduler.resume_schedule.await_args_list]
    assert resumed == ["a", "b"]


@pytest.mark.asyncio
async def test_a_lost_answer_for_a_job_its_owner_paused_leaves_it_paused(scheduler):
    scheduler.get_execution_schedules.return_value = [MagicMock(id="a")]
    scheduler.pause_schedule.side_effect = _fail_on("b", TimeoutError("no answer"))

    assert await jobs.pause_jobs(_USER, ["a", "b"]) is None
    scheduler.resume_schedule.assert_awaited_once_with("a", user_id=_USER)


@pytest.mark.asyncio
async def test_unknown_jobs_are_not_paused(scheduler):
    scheduler.get_execution_schedules.side_effect = RuntimeError("scheduler down")

    assert await jobs.pause_jobs(_USER, ["a"]) is None
    scheduler.pause_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_resume_that_fails_pauses_again_what_it_resumed(scheduler):
    """And the job it failed on, whose resume may have landed with its answer
    lost: the chat stays listed as paused, so none of its jobs may fire."""
    scheduler.resume_schedule.side_effect = _fail_on("b", RuntimeError("down"))

    with pytest.raises(RuntimeError):
        await jobs.resume_jobs(_USER, ["a", "b", "c"])

    paused = [call.args[0] for call in scheduler.pause_schedule.await_args_list]
    assert paused == ["a", "b"]


@pytest.mark.asyncio
async def test_a_resume_skips_a_job_deleted_since(scheduler):
    scheduler.resume_schedule.side_effect = _fail_on("a", NotFoundError("gone"))

    await jobs.resume_jobs(_USER, ["a", "b"])

    scheduler.pause_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_jobs_are_paused_while_none_of_them_is_listed(scheduler):
    assert await jobs.jobs_paused(_USER, ["a"]) is True

    scheduler.get_execution_schedules.return_value = [MagicMock(id="a")]
    assert await jobs.jobs_paused(_USER, ["a", "b"]) is False


@pytest.mark.asyncio
async def test_an_unreadable_scheduler_counts_as_paused(scheduler):
    scheduler.get_execution_schedules.side_effect = RuntimeError("down")

    assert await jobs.jobs_paused(_USER, ["a"]) is True
