"""The onboarding interview's session id on the user's onboarding row.

Needs the database (``server`` starts the test stack): what is tested is the
one-statement claim, which only the database can decide between racing
requests.
"""

import asyncio
import uuid

import pytest
from prisma.models import User, UserOnboarding

from backend.copilot.db import create_chat_session, delete_chat_session
from backend.copilot.model import ONBOARDING_INTERVIEW_KIND, ChatSessionMetadata
from backend.data.onboarding import (
    claim_onboarding_interview_session_id,
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


async def test_two_first_opens_at_once_record_one_interview(new_user: str):
    offered = [str(uuid.uuid4()) for _ in range(5)]

    claimed = await asyncio.gather(
        *(claim_onboarding_interview_session_id(new_user, id_) for id_ in offered)
    )

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


async def test_a_stale_id_is_replaced_once_however_many_replace_it(new_user: str):
    stale = await claim_onboarding_interview_session_id(new_user, "stale-chat")

    replaced = await asyncio.gather(
        *(
            reclaim_onboarding_interview_session_id(new_user, stale, f"fresh-{n}")
            for n in range(5)
        )
    )

    assert len(set(replaced)) == 1
    assert replaced[0] != stale
    assert await get_onboarding_interview_session_id(new_user) == replaced[0]
