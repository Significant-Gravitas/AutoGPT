"""The typed interview's details: read once per turn of the person's, kept,
and never read before they have said anything."""

from unittest.mock import AsyncMock

import pytest
from fakeredis.aioredis import FakeRedis
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import interview_details
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.onboarding_interview_details import DetailsRead, InterviewDetails


def _session(*said: str) -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False)
    messages = [ChatMessage(role="assistant", content="What do you do?")]
    for text in said:
        messages.append(ChatMessage(role="user", content=text))
        messages.append(ChatMessage(role="assistant", content="Tell me more."))
    return session.model_copy(update={"messages": messages})


@pytest.fixture
def reads(mocker: MockerFixture) -> AsyncMock:
    mocker.patch.object(
        interview_details.settings.secrets, "gemini_api_key", "test-key"
    )
    redis = FakeRedis(decode_responses=True)
    mocker.patch.object(
        interview_details, "get_redis_async", new=AsyncMock(return_value=redis)
    )
    return mocker.patch.object(
        interview_details,
        "extract_details",
        new=AsyncMock(
            return_value=DetailsRead(
                details=InterviewDetails(business="Bakery"),
                tokens_in=800,
                tokens_out=90,
                latency_s=0.5,
            )
        ),
    )


@pytest.mark.asyncio
async def test_a_turn_already_read_answers_from_the_cache(reads: AsyncMock):
    session = _session("I run a bakery")

    first = await interview_details.read_interview_details(session)
    again = await interview_details.read_interview_details(session)

    assert first == again == InterviewDetails(business="Bakery")
    reads.assert_awaited_once()
    lines = reads.await_args.kwargs["lines"]
    assert lines == [
        ("otto", "What do you do?"),
        ("user", "I run a bakery"),
        ("otto", "Tell me more."),
    ]


@pytest.mark.asyncio
async def test_a_new_turn_is_read_again(reads: AsyncMock):
    await interview_details.read_interview_details(_session("I run a bakery"))
    await interview_details.read_interview_details(
        _session("I run a bakery", "Just me and Sam")
    )

    assert reads.await_count == 2


@pytest.mark.asyncio
async def test_nothing_is_read_before_the_person_has_said_anything(reads: AsyncMock):
    assert await interview_details.read_interview_details(_session()) is None
    reads.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_read_that_fails_is_none_and_not_kept(reads: AsyncMock):
    reads.side_effect = RuntimeError("vendor down")
    session = _session("I run a bakery")

    assert await interview_details.read_interview_details(session) is None
    reads.side_effect = None
    assert await interview_details.read_interview_details(session) is not None
