"""Otto starting the user's first suggested tasks, against Postgres.

``kickoff_test.py`` fakes the row and every chat call; this runs a stored
take through the real chat creation, the real checklist write and the real
greeting, and stops only where the turn would be handed to the executor
queue.
"""

import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest_asyncio
from prisma import Json
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import ChatSession, OnboardingBrainDump, User
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import db, kickoff, service
from backend.copilot.db import user_has_any_session
from backend.copilot.model import ONBOARDING_DRAFT_KIND

RECORDING_ID = "rec-kickoff"
GREETING = "Let's get your invoices handled."
STORED = [
    {"title": "Chase unpaid invoices", "prompt": "Chase my invoices.", "icon": "bell"},
    {"title": "Draft the newsletter", "prompt": "Draft it.", "icon": "megaphone"},
    {"title": "Watch competitors", "prompt": "Watch them.", "icon": "eye"},
]


@pytest_asyncio.fixture(loop_scope="session")
async def user_id(server) -> AsyncGenerator[str, None]:
    user_id = str(uuid.uuid4())
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    yield user_id
    await User.prisma().delete(where={"id": user_id})


async def _store_finished_take(user_id: str) -> None:
    await db.start_dump(user_id, RECORDING_ID, BrainDumpInputMode.typed)
    await OnboardingBrainDump.prisma().update(
        where={"userId": user_id},
        data={
            "status": BrainDumpStatus.completed,
            "transcript": "I run a bakery and chase invoices every Monday.",
            "greeting": GREETING,
            "suggestedPrompts": Json(STORED),
            "recommendedExperts": Json({"experts": [], "source": "fallback"}),
        },
    )


async def test_a_finished_take_starts_two_drafts_and_keeps_the_greeting(
    user_id: str, mocker: MockerFixture
):
    await _store_finished_take(user_id)
    # Admission (plan, bans, usage) is kickoff_test's to cover.
    mocker.patch.object(kickoff, "turn_refusal", new=AsyncMock(return_value=None))
    # The executor queue is the one thing not run for real.
    dispatched = mocker.patch(
        "backend.copilot.executor.utils.dispatch_turn", new=AsyncMock()
    )
    # The package's conftest fakes the session lookup the greeting makes.
    mocker.patch.object(service, "user_has_any_session", new=user_has_any_session)

    response = await kickoff.kickoff_suggested_tasks(user_id)

    assert (response.status, response.started_task_ids) == ("started", ["0", "1"])
    draft_ids = [
        kickoff.draft_session_id(user_id, RECORDING_ID, task_id)
        for task_id in ("0", "1")
    ]
    chats = await ChatSession.prisma().find_many(where={"userId": user_id})
    assert sorted(chat.id for chat in chats) == sorted(draft_ids)
    assert all(
        isinstance(chat.metadata, dict)
        and chat.metadata.get("kind") == ONBOARDING_DRAFT_KIND
        for chat in chats
    )
    assert [
        (call.kwargs["session_id"], call.kwargs["message"])
        for call in dispatched.await_args_list
    ] == [(draft_ids[0], "Chase my invoices."), (draft_ids[1], "Draft it.")]
    dump = await db.get_dump(user_id)
    assert dump is not None
    stored = dump.suggestedPrompts
    assert isinstance(stored, list)
    assert [
        (task.get("state"), task.get("kickoff"), task.get("session_id"))
        for task in stored
    ] == [
        ("sent", True, draft_ids[0]),
        ("sent", True, draft_ids[1]),
        ("open", False, None),
    ]
    intro = await service.get_intro_card(user_id)
    assert (intro.path, intro.greeting, intro.greeting_done) == ("A", GREETING, False)
