"""The suggested-task writes against Postgres.

The route tests fake the row; whether a stale read really loses depends on
Prisma moving ``updatedAt`` on every write, which only the database shows.
"""

import uuid
from collections.abc import AsyncGenerator

import pytest_asyncio
from prisma.models import OnboardingBrainDump, User

from backend.api.features.onboarding_dump import db, suggested_tasks
from backend.api.features.onboarding_dump.models import SuggestedTasksUpdate


@pytest_asyncio.fixture(loop_scope="session")
async def user_id(server) -> AsyncGenerator[str, None]:
    user_id = str(uuid.uuid4())
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    yield user_id
    await User.prisma().delete(where={"id": user_id})


async def test_a_write_from_a_stale_read_hits_nothing(user_id: str):
    row = await db.ensure_dump(user_id)

    assert await db.replace_suggested_prompts(user_id, row.updatedAt, [{"id": "a"}])
    assert not await db.replace_suggested_prompts(user_id, row.updatedAt, [{"id": "b"}])

    stored = await OnboardingBrainDump.prisma().find_unique(where={"userId": user_id})
    assert stored is not None
    assert stored.suggestedPrompts == [{"id": "a"}]


async def test_ensure_dump_leaves_an_existing_row_alone(user_id: str):
    first = await db.ensure_dump(user_id)
    await db.replace_suggested_prompts(user_id, first.updatedAt, [{"id": "a"}])

    again = await db.ensure_dump(user_id)

    assert again.id == first.id
    assert again.suggestedPrompts == [{"id": "a"}]
    assert again.greetingSeen is False


async def test_two_changes_in_a_row_both_land(user_id: str):
    await suggested_tasks.update_suggested_tasks(
        user_id, SuggestedTasksUpdate(task_ids=["0"], state="dismissed")
    )
    await suggested_tasks.update_suggested_tasks(
        user_id,
        SuggestedTasksUpdate(task_ids=["1"], state="sent", session_id="session-1"),
    )

    tasks = (await suggested_tasks.get_suggested_tasks(user_id)).tasks
    assert [task.state for task in tasks][:3] == ["dismissed", "sent", "open"]
    assert tasks[1].session_id == "session-1"
