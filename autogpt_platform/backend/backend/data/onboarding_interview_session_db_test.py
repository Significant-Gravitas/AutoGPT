"""The onboarding interview's session id on the user's onboarding row.

Needs the database (``server`` starts the test stack): what is tested is the
one-statement claim, which only the database can decide between racing
requests. The races are forced, not left to timing: racing writes wait for
each other, and a first write of the onboarding row can lose to another's.
"""

import asyncio
import uuid

import pytest
from prisma.actions import UserOnboardingActions
from prisma.models import User, UserOnboarding

from backend.copilot.db import create_chat_session, delete_chat_session
from backend.copilot.model import ONBOARDING_INTERVIEW_KIND, ChatSessionMetadata
from backend.data.onboarding import (
    claim_onboarding_interview_session_id,
    ensure_user_onboarding,
    get_onboarding_interview_session_id,
    reclaim_onboarding_interview_session_id,
)
from backend.util.test import SpinTestServer


@pytest.fixture
async def new_user(server: SpinTestServer):
    user_id = f"onboarding-interview-{uuid.uuid4()}"
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    yield user_id
    await User.prisma().delete_many(where={"id": user_id})


async def test_two_first_writes_at_once_both_get_the_row_one_of_them_created(
    new_user: str, monkeypatch: pytest.MonkeyPatch
):
    _the_first_insert_loses_the_race(monkeypatch)

    async with asyncio.TaskGroup() as writes:
        losing = writes.create_task(ensure_user_onboarding(new_user))
        winning = writes.create_task(ensure_user_onboarding(new_user))

    assert losing.result().id == winning.result().id
    assert await UserOnboarding.prisma().count(where={"userId": new_user}) == 1


async def test_two_first_opens_at_once_record_one_interview(
    new_user: str, monkeypatch: pytest.MonkeyPatch
):
    offered = [str(uuid.uuid4()) for _ in range(5)]
    _the_first_insert_loses_the_race(monkeypatch)
    _every_claim_waits_for_the_others(monkeypatch, parties=len(offered))

    async with asyncio.TaskGroup() as opens:
        claims = [
            opens.create_task(claim_onboarding_interview_session_id(new_user, id_))
            for id_ in offered
        ]

    claimed = [claim.result() for claim in claims]
    assert len(set(claimed)) == 1
    assert claimed[0] in offered
    assert await get_onboarding_interview_session_id(new_user) == claimed[0]


async def test_a_recorded_interview_is_kept_by_every_later_claim(new_user: str):
    first = await claim_onboarding_interview_session_id(new_user, "interview-a")

    again = await claim_onboarding_interview_session_id(new_user, "interview-b")

    assert first == again == "interview-a"


async def test_no_interview_is_recorded_until_one_is_opened(new_user: str):
    assert await get_onboarding_interview_session_id(new_user) is None

    await UserOnboarding.prisma().create(data={"userId": new_user})

    assert await get_onboarding_interview_session_id(new_user) is None


async def test_the_claim_creates_the_onboarding_row_it_needs(new_user: str):
    assert await UserOnboarding.prisma().find_unique(where={"userId": new_user}) is None

    claimed = await claim_onboarding_interview_session_id(new_user, "interview-a")

    row = await UserOnboarding.prisma().find_unique(where={"userId": new_user})
    assert row is not None
    assert row.interviewSessionId == claimed == "interview-a"


async def test_deleting_the_interview_lets_the_next_start_record_a_new_one(
    new_user: str,
):
    """A new interview is a new take: its id, and so its take's id, is new."""
    old = await claim_onboarding_interview_session_id(new_user, str(uuid.uuid4()))
    await create_chat_session(
        old, new_user, metadata=ChatSessionMetadata(kind=ONBOARDING_INTERVIEW_KIND)
    )

    assert await delete_chat_session(old, new_user)

    assert await get_onboarding_interview_session_id(new_user) is None
    fresh = await claim_onboarding_interview_session_id(new_user, str(uuid.uuid4()))
    assert fresh != old


async def test_deleting_another_chat_keeps_the_interview_recorded(new_user: str):
    interview = await claim_onboarding_interview_session_id(new_user, str(uuid.uuid4()))
    other = str(uuid.uuid4())
    await create_chat_session(other, new_user)

    assert await delete_chat_session(other, new_user)

    assert await get_onboarding_interview_session_id(new_user) == interview


async def test_a_stale_id_is_replaced_once_however_many_replace_it(
    new_user: str, monkeypatch: pytest.MonkeyPatch
):
    stale = await claim_onboarding_interview_session_id(new_user, "stale-chat")
    fresh = [f"fresh-{n}" for n in range(5)]
    _every_claim_waits_for_the_others(monkeypatch, parties=len(fresh))

    async with asyncio.TaskGroup() as replacements:
        replacing = [
            replacements.create_task(
                reclaim_onboarding_interview_session_id(new_user, stale, id_)
            )
            for id_ in fresh
        ]

    replaced = [replacement.result() for replacement in replacing]
    assert len(set(replaced)) == 1
    assert replaced[0] in fresh
    assert await get_onboarding_interview_session_id(new_user) == replaced[0]


def _the_first_insert_loses_the_race(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first write of the onboarding row finds none, and inserts only once
    another first write's row is in, so the database refuses it as it refuses
    the losing write of a real race."""
    upsert = UserOnboardingActions.upsert
    other_row_in = asyncio.Event()
    calls = 0

    async def racing_upsert(self, where, data, include=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            async with asyncio.timeout(30):
                await other_row_in.wait()
            return await UserOnboarding.prisma().create(data=data["create"])
        try:
            return await upsert(self, where=where, data=data, include=include)
        finally:
            other_row_in.set()

    monkeypatch.setattr(UserOnboardingActions, "upsert", racing_upsert)


def _every_claim_waits_for_the_others(
    monkeypatch: pytest.MonkeyPatch, parties: int
) -> None:
    """Each racer writes the column only once every racer is about to, so all
    of them decide on it at once, after all have read whatever they read."""
    barrier = asyncio.Barrier(parties)
    update_many = UserOnboardingActions.update_many

    async def at_once(self, data, where):
        async with asyncio.timeout(30):
            await barrier.wait()
        return await update_many(self, data=data, where=where)

    monkeypatch.setattr(UserOnboardingActions, "update_many", at_once)
