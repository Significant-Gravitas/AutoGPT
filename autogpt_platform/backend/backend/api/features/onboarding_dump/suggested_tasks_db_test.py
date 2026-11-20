"""The suggested-task writes against Postgres.

The route tests fake the row; whether a stale read really loses depends on
Prisma moving ``updatedAt`` on every write, which only the database shows.
"""

import uuid
from collections.abc import AsyncGenerator

import pytest_asyncio
from prisma.enums import BrainDumpInputMode
from prisma.models import OnboardingBrainDump, User

from backend.api.features.onboarding_dump import db, suggested_tasks
from backend.api.features.onboarding_dump.models import (
    ExpertRecommendations,
    RecommendedExpert,
    SuggestedTasksUpdate,
)

# The greeting's prompts as its job stores them: no ids, no states.
GREETING_PROMPTS = [
    {"title": "Chase invoices", "prompt": "Chase them.", "icon": "currency-dollar"},
    {"title": "Draft posts", "prompt": "Draft.", "icon": "megaphone"},
    {"title": "Plan my week", "prompt": "Plan it.", "icon": "calendar-check"},
]

# The same prompts as the plan job stores them once ranked: each row keeps
# the id its place in the greeting's list gave it.
PLAN_ROWS = [
    {
        "title": "Draft posts",
        "prompt": "Draft.",
        "icon": "megaphone",
        "expert_template_id": "tmpl-marketing",
        "why": "posts keep slipping",
        "id": "1",
    },
    {
        "title": "Plan my week",
        "prompt": "Plan it.",
        "icon": "calendar-check",
        "expert_template_id": None,
        "why": None,
        "id": "2",
    },
    {
        "title": "Chase invoices",
        "prompt": "Chase them.",
        "icon": "currency-dollar",
        "expert_template_id": None,
        "why": "invoices go unpaid",
        "id": "0",
    },
]

PLAN_TEAM = ExpertRecommendations(
    experts=[
        RecommendedExpert(
            template_id="tmpl-marketing", name="Mia", role="Marketing", reason="Posts"
        )
    ],
    source="llm",
).model_dump(mode="json")


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


async def test_prompts_stored_before_owners_existed_still_rehydrate(user_id: str):
    row = await db.ensure_dump(user_id)
    await db.replace_suggested_prompts(
        user_id,
        row.updatedAt,
        [
            {"title": "Chase invoices", "prompt": "Chase them.", "icon": "bell"},
            {"title": "Draft posts", "prompt": "Draft.", "state": "sent"},
        ],
    )

    tasks = (await suggested_tasks.get_suggested_tasks(user_id)).tasks

    assert [task.title for task in tasks] == ["Chase invoices", "Draft posts"]
    assert all(task.expert_template_id is None for task in tasks)
    assert all(task.why is None and task.expert_id is None for task in tasks)
    assert [task.state for task in tasks] == ["open", "sent"]


async def test_a_plan_from_a_fresh_read_stores_its_tasks_and_team_together(
    user_id: str,
):
    take = await db.start_dump(user_id, "rec-1", BrainDumpInputMode.voice)

    assert await db.replace_plan(user_id, "rec-1", take.updatedAt, PLAN_ROWS, PLAN_TEAM)

    stored = await db.get_dump(user_id)
    assert stored is not None
    assert stored.suggestedPrompts == PLAN_ROWS
    assert stored.recommendedExperts == PLAN_TEAM
    assert stored.updatedAt > take.updatedAt


async def test_a_plan_from_a_stale_read_loses_to_the_write_that_landed_first(
    user_id: str,
):
    take = await db.start_dump(user_id, "rec-1", BrainDumpInputMode.voice)
    dismissed = [{**GREETING_PROMPTS[0], "state": "dismissed"}]
    assert await db.replace_suggested_prompts(user_id, take.updatedAt, dismissed)

    assert not await db.replace_plan(
        user_id, "rec-1", take.updatedAt, PLAN_ROWS, PLAN_TEAM
    )

    stored = await db.get_dump(user_id)
    assert stored is not None
    assert stored.suggestedPrompts == dismissed
    assert stored.recommendedExperts is None


async def test_a_plan_for_a_take_that_was_replaced_writes_nothing(user_id: str):
    await db.start_dump(user_id, "rec-1", BrainDumpInputMode.voice)
    new_take = await db.start_dump(user_id, "rec-2", BrainDumpInputMode.voice)

    assert not await db.replace_plan(
        user_id, "rec-1", new_take.updatedAt, PLAN_ROWS, PLAN_TEAM
    )

    stored = await db.get_dump(user_id)
    assert stored is not None
    assert stored.recordingId == "rec-2"
    assert stored.suggestedPrompts == []
    assert stored.recommendedExperts is None
    assert stored.updatedAt == new_take.updatedAt


async def test_a_ranked_plan_keeps_the_ids_its_tasks_were_handed_out_with(
    user_id: str,
):
    take = await db.start_dump(user_id, "rec-1", BrainDumpInputMode.voice)
    await db.replace_suggested_prompts(user_id, take.updatedAt, GREETING_PROMPTS)
    handed_out = (await suggested_tasks.get_suggested_tasks(user_id)).tasks
    id_of = {task.title: task.id for task in handed_out}
    greeted = await db.get_dump(user_id)
    assert greeted is not None

    assert await db.replace_plan(
        user_id, "rec-1", greeted.updatedAt, PLAN_ROWS, PLAN_TEAM
    )

    stored = await db.get_dump(user_id)
    assert stored is not None
    ranked = suggested_tasks.stored_tasks(stored.suggestedPrompts)
    assert [(task.title, task.id) for task in ranked] == [
        ("Draft posts", id_of["Draft posts"]),
        ("Plan my week", id_of["Plan my week"]),
        ("Chase invoices", id_of["Chase invoices"]),
    ]
    after = await suggested_tasks.update_suggested_tasks(
        user_id,
        SuggestedTasksUpdate(task_ids=[id_of["Chase invoices"]], state="dismissed"),
    )
    assert [(task.title, task.state) for task in after.tasks] == [
        ("Draft posts", "open"),
        ("Plan my week", "open"),
        ("Chase invoices", "dismissed"),
    ]


async def test_the_greeting_writes_its_tasks_onto_a_take_with_none(user_id: str):
    await db.start_dump(user_id, "rec-1", BrainDumpInputMode.voice)

    assert await db.store_greeting(user_id, "rec-1", "Hi.", GREETING_PROMPTS)

    stored = await db.get_dump(user_id)
    assert stored is not None
    assert (stored.greeting, stored.suggestedPrompts) == ("Hi.", GREETING_PROMPTS)


async def test_the_greeting_leaves_tasks_the_step_settled_on_alone(user_id: str):
    take = await db.start_dump(user_id, "rec-1", BrainDumpInputMode.voice)
    settled = [{**row, "state": "open"} for row in PLAN_ROWS]
    assert await db.replace_plan(user_id, "rec-1", take.updatedAt, settled, PLAN_TEAM)

    assert await db.store_greeting(user_id, "rec-1", "Hi.", GREETING_PROMPTS)

    stored = await db.get_dump(user_id)
    assert stored is not None
    assert (stored.greeting, stored.suggestedPrompts) == ("Hi.", settled)


async def test_a_greeting_for_a_replaced_take_writes_nothing(user_id: str):
    await db.start_dump(user_id, "rec-1", BrainDumpInputMode.voice)
    await db.start_dump(user_id, "rec-2", BrainDumpInputMode.voice)

    assert not await db.store_greeting(user_id, "rec-1", "Hi.", GREETING_PROMPTS)

    stored = await db.get_dump(user_id)
    assert stored is not None
    assert (stored.greeting, stored.suggestedPrompts) == (None, [])


async def test_a_team_from_a_fresh_read_is_stored(user_id: str):
    take = await db.start_dump(user_id, "rec-1", BrainDumpInputMode.voice)

    assert await db.replace_team(user_id, "rec-1", take.updatedAt, PLAN_TEAM)

    stored = await db.get_dump(user_id)
    assert stored is not None
    assert stored.recommendedExperts == PLAN_TEAM


async def test_a_team_from_a_stale_read_hits_nothing(user_id: str):
    take = await db.start_dump(user_id, "rec-1", BrainDumpInputMode.voice)
    assert await db.store_greeting(user_id, "rec-1", "Hi.", GREETING_PROMPTS)

    assert not await db.replace_team(user_id, "rec-1", take.updatedAt, PLAN_TEAM)

    stored = await db.get_dump(user_id)
    assert stored is not None
    assert stored.recommendedExperts is None
