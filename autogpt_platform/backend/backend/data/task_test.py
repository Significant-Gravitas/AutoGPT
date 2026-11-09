"""Integration tests for the task data layer, against the test database."""

import logging
from uuid import uuid4

import pytest
from prisma.enums import TaskCreator, TaskStatus
from prisma.models import ChatSession, Expert, Task, User

from backend.data import task as task_db
from backend.data.task import TaskPatch
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)


async def _user() -> str:
    user_id = f"task-{uuid4()}"
    await User.prisma().create(
        data={"id": user_id, "email": f"{user_id}@example.com", "name": "Task Test"}
    )
    return user_id


async def _expert(user_id: str) -> str:
    expert = await Expert.prisma().create(
        data={
            "ownerUserId": user_id,
            "name": "Builder",
            "role": "developer",
            "identity": "test",
        }
    )
    return expert.id


async def _chat(user_id: str) -> str:
    chat = await ChatSession.prisma().create(data={"userId": user_id})
    return chat.id


async def _cleanup(*user_ids: str) -> None:
    for user_id in user_ids:
        try:
            await Task.prisma().delete_many(where={"userId": user_id})
            await ChatSession.prisma().delete_many(where={"userId": user_id})
            await Expert.prisma().delete_many(where={"ownerUserId": user_id})
            await User.prisma().delete_many(where={"id": user_id})
        except Exception as exc:
            logger.warning("cleanup for %s failed: %s", user_id, exc)


@pytest.mark.asyncio(loop_scope="session")
async def test_one_chat_carries_several_tasks_through_their_lifecycle(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        chat_id = await _chat(user_id)
        expert_id = await _expert(user_id)

        build = await task_db.create_task(
            user_id=user_id,
            title="  Build   feature X ",
            created_by=TaskCreator.OTTO,
            owner_expert_id=expert_id,
            session_id=chat_id,
        )
        announce = await task_db.create_task(
            user_id=user_id,
            title="Announce feature X",
            created_by=TaskCreator.USER,
            session_id=chat_id,
        )

        assert build.title == "Build feature X"
        assert build.status == TaskStatus.TODO
        assert build.owner_expert_id == expert_id
        assert build.session_ids == [chat_id]
        assert announce.owner_expert_id is None

        in_chat = await task_db.list_tasks(user_id, session_id=chat_id)
        assert {t.id for t in in_chat} == {build.id, announce.id}

        started = await task_db.update_task(
            build.id, user_id, TaskPatch(status=TaskStatus.IN_PROGRESS)
        )
        assert started.started_at is not None and started.completed_at is None

        done = await task_db.update_task(
            build.id, user_id, TaskPatch(status=TaskStatus.DONE)
        )
        assert done.completed_at is not None
        assert done.started_at == started.started_at

        reopened = await task_db.update_task(
            build.id, user_id, TaskPatch(status=TaskStatus.TODO)
        )
        assert reopened.started_at is None and reopened.completed_at is None

        assert await task_db.archive_task(announce.id, user_id) is True
        assert await task_db.get_task(announce.id, user_id) is None
        assert [t.id for t in await task_db.list_tasks(user_id)] == [build.id]
        assert await task_db.archive_task(announce.id, user_id) is False
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_task_spans_several_chats_and_links_are_idempotent(
    server: SpinTestServer,
):
    user_id = await _user()
    try:
        first, second = await _chat(user_id), await _chat(user_id)
        task = await task_db.create_task(
            user_id=user_id,
            title="Ship it",
            created_by=TaskCreator.OTTO,
            session_id=first,
        )

        linked = await task_db.attach_task_session(task.id, second, user_id)
        again = await task_db.attach_task_session(task.id, second, user_id)
        assert set(linked.session_ids) == {first, second}
        assert set(again.session_ids) == {first, second}

        unlinked = await task_db.detach_task_session(task.id, first, user_id)
        assert unlinked.session_ids == [second]
        assert await task_db.list_tasks(user_id, session_id=first) == []
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_owner_changes_and_description_clearing(server: SpinTestServer):
    user_id = await _user()
    try:
        expert_id = await _expert(user_id)
        task = await task_db.create_task(
            user_id=user_id,
            title="Write copy",
            description="For the changelog",
            created_by=TaskCreator.OTTO,
        )

        assigned = await task_db.update_task(
            task.id, user_id, TaskPatch(owner_expert_id=expert_id)
        )
        assert assigned.owner_expert_id == expert_id
        assert assigned.description == "For the changelog"

        back = await task_db.update_task(
            task.id, user_id, TaskPatch(assign_to_otto=True, description="")
        )
        assert back.owner_expert_id is None
        assert back.description is None
        assert back.title == "Write copy"
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_another_users_task_expert_and_chat_read_as_not_found(
    server: SpinTestServer,
):
    owner, stranger = await _user(), await _user()
    try:
        their_chat = await _chat(owner)
        their_expert = await _expert(owner)
        task = await task_db.create_task(
            user_id=owner, title="Private", created_by=TaskCreator.USER
        )

        assert await task_db.get_task(task.id, stranger) is None
        assert await task_db.list_tasks(stranger) == []
        assert await task_db.archive_task(task.id, stranger) is False
        with pytest.raises(NotFoundError):
            await task_db.update_task(task.id, stranger, TaskPatch(title="Mine"))
        with pytest.raises(NotFoundError):
            await task_db.attach_task_session(task.id, their_chat, stranger)

        mine = await task_db.create_task(
            user_id=stranger, title="Mine", created_by=TaskCreator.USER
        )
        with pytest.raises(NotFoundError):
            await task_db.attach_task_session(mine.id, their_chat, stranger)
        with pytest.raises(NotFoundError):
            await task_db.update_task(
                mine.id, stranger, TaskPatch(owner_expert_id=their_expert)
            )
        with pytest.raises(NotFoundError):
            await task_db.create_task(
                user_id=stranger,
                title="Steal",
                created_by=TaskCreator.USER,
                session_id=their_chat,
            )
        assert [t.id for t in await task_db.list_tasks(stranger)] == [mine.id]
    finally:
        await _cleanup(owner, stranger)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("title", ["", "   ", "x" * (task_db.TITLE_MAX_CHARS + 1)])
async def test_a_bad_title_is_refused_and_nothing_is_written(
    server: SpinTestServer, title: str
):
    user_id = await _user()
    try:
        with pytest.raises(ValueError):
            await task_db.create_task(
                user_id=user_id, title=title, created_by=TaskCreator.USER
            )
        assert await task_db.list_tasks(user_id) == []
    finally:
        await _cleanup(user_id)
