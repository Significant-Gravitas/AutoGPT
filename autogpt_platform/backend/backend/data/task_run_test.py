"""Integration tests for a task run started from the task's page, against the
test database: claiming a run, undoing it, and settling it when its chat's
turn ends."""

import logging
from uuid import uuid4

import pytest
from prisma.enums import TaskCreator, TaskStatus
from prisma.models import ChatSession, Task, User

from backend.data import task as task_db
from backend.data import task_record as record_db
from backend.data import task_run as run_db
from backend.data.task import TaskPatch
from backend.util.exceptions import ConflictError, NotFoundError
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)


async def _user() -> str:
    user_id = f"task-run-{uuid4()}"
    await User.prisma().create(
        data={"id": user_id, "email": f"{user_id}@example.com", "name": "Run Test"}
    )
    return user_id


async def _chat(user_id: str) -> str:
    return (await ChatSession.prisma().create(data={"userId": user_id})).id


async def _cleanup(*user_ids: str) -> None:
    for user_id in user_ids:
        try:
            await Task.prisma().delete_many(where={"userId": user_id})
            await ChatSession.prisma().delete_many(where={"userId": user_id})
            await User.prisma().delete_many(where={"id": user_id})
        except Exception as exc:
            logger.warning("cleanup for %s failed: %s", user_id, exc)


async def _task(user_id: str, status: TaskStatus = TaskStatus.TODO) -> task_db.Task:
    task = await task_db.create_task(
        user_id=user_id, title="Plan the launch", created_by=TaskCreator.USER
    )
    if status != TaskStatus.TODO:
        task = await task_db.update_task(task.id, user_id, TaskPatch(status=status))
    return task


async def _running(user_id: str) -> tuple[task_db.Task, str]:
    """A task claimed for a run, worked on in its own chat."""
    task = await _task(user_id)
    await run_db.claim_task_run(task.id, user_id)
    chat_id = await _chat(user_id)
    task = await task_db.update_task(
        task.id, user_id, TaskPatch(work_session_id=chat_id)
    )
    return task, chat_id


@pytest.mark.asyncio(loop_scope="session")
async def test_claim_takes_a_suggested_task_into_progress(server: SpinTestServer):
    user_id = await _user()
    try:
        task = await _task(user_id)
        await Task.prisma().update(where={"id": task.id}, data={"isSuggested": True})

        claim = await run_db.claim_task_run(task.id, user_id)

        assert claim.task.status == TaskStatus.IN_PROGRESS
        assert claim.task.is_suggested is False
        assert claim.task.started_at is not None
        assert claim.retry is False
        assert claim.before.status == TaskStatus.TODO
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("status", [TaskStatus.FAILED, TaskStatus.CANCELLED])
async def test_claiming_a_failed_or_cancelled_task_counts_another_attempt(
    server: SpinTestServer, status: TaskStatus
):
    user_id = await _user()
    try:
        task = await _task(user_id, status)
        await Task.prisma().update(
            where={"id": task.id}, data={"failureReason": "It broke"}
        )

        claim = await run_db.claim_task_run(task.id, user_id)

        assert claim.retry is True
        assert claim.task.attempt == 2
        assert claim.task.failure_reason is None
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("status", [TaskStatus.IN_PROGRESS, TaskStatus.NEEDS_INPUT])
async def test_a_task_being_worked_on_cannot_be_claimed(
    server: SpinTestServer, status: TaskStatus
):
    user_id = await _user()
    try:
        task = await _task(user_id, status)

        with pytest.raises(ConflictError):
            await run_db.claim_task_run(task.id, user_id)
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_another_users_task_is_not_found(server: SpinTestServer):
    owner, other = await _user(), await _user()
    try:
        task = await _task(owner)

        with pytest.raises(NotFoundError):
            await run_db.claim_task_run(task.id, other)
    finally:
        await _cleanup(owner, other)


@pytest.mark.asyncio(loop_scope="session")
async def test_undo_puts_the_task_back_as_it_was(server: SpinTestServer):
    user_id = await _user()
    try:
        task = await _task(user_id, TaskStatus.FAILED)
        claim = await run_db.claim_task_run(task.id, user_id)

        await run_db.undo_task_run(claim, user_id)

        after = await task_db.get_task(task.id, user_id)
        assert after is not None
        assert after.status == TaskStatus.FAILED
        assert after.attempt == 1
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_finished_worker_settles_the_task_done(server: SpinTestServer):
    user_id = await _user()
    try:
        task, chat_id = await _running(user_id)
        await Task.prisma().update(
            where={"id": task.id}, data={"result": "Launch plan written."}
        )

        status = await run_db.settle_task_run(task.id, chat_id, user_id, "completed")

        after = await task_db.get_task(task.id, user_id)
        assert status == TaskStatus.DONE
        assert after is not None and after.status == TaskStatus.DONE
        assert after.result == "Launch plan written."
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "ending,reason",
    [
        ("completed", run_db.NO_RESULT_REASON),
        ("failed", "The work stopped before it finished."),
        ("timed_out", "The work ran out of time."),
    ],
)
async def test_a_turn_that_ends_without_a_result_fails_the_task(
    server: SpinTestServer, ending: run_db.TurnEnding, reason: str
):
    user_id = await _user()
    try:
        task, chat_id = await _running(user_id)

        status = await run_db.settle_task_run(task.id, chat_id, user_id, ending)

        record = await record_db.get_task_record(task.id, user_id)
        assert status == TaskStatus.FAILED
        assert record is not None
        assert record.task.failure_reason == reason
        assert record.activity[0].text == f"Did not finish: {reason}"
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_stopped_turn_cancels_the_task(server: SpinTestServer):
    user_id = await _user()
    try:
        task, chat_id = await _running(user_id)

        status = await run_db.settle_task_run(task.id, chat_id, user_id, "cancelled")

        assert status == TaskStatus.CANCELLED
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_task_waiting_on_the_user_or_on_subtasks_stays_open(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, chat_id = await _running(user_id)
        await task_db.create_task(
            user_id=user_id,
            title="Draft the email",
            created_by=TaskCreator.OTTO,
            parent_task_id=task.id,
        )
        assert (
            await run_db.settle_task_run(task.id, chat_id, user_id, "completed") is None
        )

        await task_db.update_task(
            task.id, user_id, TaskPatch(status=TaskStatus.NEEDS_INPUT)
        )
        assert await run_db.settle_task_run(task.id, chat_id, user_id, "failed") is None
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_an_older_runs_chat_does_not_settle_a_newer_run(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        task, _ = await _running(user_id)
        old_chat = await _chat(user_id)

        status = await run_db.settle_task_run(task.id, old_chat, user_id, "failed")

        after = await task_db.get_task(task.id, user_id)
        assert status is None
        assert after is not None and after.status == TaskStatus.IN_PROGRESS
    finally:
        await _cleanup(user_id)
