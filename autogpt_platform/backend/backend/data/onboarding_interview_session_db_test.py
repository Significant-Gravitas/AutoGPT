"""The onboarding interview's session id on the user's onboarding row.

Needs the database (``server`` starts the test stack): what is tested is the
one-statement claim, which only the database can decide between racing
requests.
"""

import asyncio
import uuid

import pytest
from prisma.models import User, UserOnboarding

from backend.data.onboarding import (
    claim_onboarding_interview_session_id,
    get_onboarding_interview_session_id,
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
