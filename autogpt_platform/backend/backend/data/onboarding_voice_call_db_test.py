"""The onboarding voice call on the user's onboarding row.

Needs the database (``server`` starts the test stack).
"""

import uuid

import pytest
from prisma.models import User, UserOnboarding

from backend.data.onboarding import (
    claim_onboarding_interview_session_id,
    get_onboarding_interview_session_id,
    get_onboarding_voice_call_id,
    record_onboarding_voice_call,
)
from backend.util.test import SpinTestServer


@pytest.fixture
async def new_user(server: SpinTestServer):
    user_id = f"onboarding-voice-{uuid.uuid4()}"
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    yield user_id
    await User.prisma().delete_many(where={"id": user_id})


async def test_no_call_is_recorded_until_one_starts(new_user: str):
    assert await get_onboarding_voice_call_id(new_user) is None


async def test_a_call_is_recorded_on_the_row_it_creates(new_user: str):
    call_id = uuid.uuid4().hex
    await record_onboarding_voice_call(new_user, call_id)
    row = await UserOnboarding.prisma().find_unique(where={"userId": new_user})
    assert row is not None and row.voiceCallId == call_id


async def test_a_later_call_replaces_the_earlier_one(new_user: str):
    first, second = uuid.uuid4().hex, uuid.uuid4().hex
    await record_onboarding_voice_call(new_user, first)
    await record_onboarding_voice_call(new_user, second)
    assert await get_onboarding_voice_call_id(new_user) == second


async def test_a_call_leaves_the_typed_interview_on_the_row_alone(new_user: str):
    interview = await claim_onboarding_interview_session_id(new_user, "interview-a")
    await record_onboarding_voice_call(new_user, uuid.uuid4().hex)
    assert await get_onboarding_interview_session_id(new_user) == interview
