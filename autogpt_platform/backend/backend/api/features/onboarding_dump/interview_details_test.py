"""The typed interview's details: the model is asked at most twice per turn of
the person's however many reads arrive, every read is kept (an empty one
too), and each call's tokens reach the platform cost log."""

import asyncio
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


def _read(details: InterviewDetails | None) -> DetailsRead:
    return DetailsRead(details=details, tokens_in=800, tokens_out=90, latency_s=0.5)


@pytest.fixture
def cost_log(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch.object(
        interview_details, "persist_and_record_usage", new=AsyncMock()
    )


@pytest.fixture
def reads(mocker: MockerFixture, cost_log: AsyncMock) -> AsyncMock:
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
        new=AsyncMock(return_value=_read(InterviewDetails(business="Bakery"))),
    )


@pytest.mark.asyncio
async def test_a_turn_already_read_answers_from_the_cache(reads: AsyncMock):
    session = _session("I run a bakery")

    first = await interview_details.read_interview_details(session)
    again = await interview_details.read_interview_details(session)

    assert first == again == InterviewDetails(business="Bakery")
    reads.assert_awaited_once()
    assert reads.await_args.kwargs["lines"] == [
        ("otto", "What do you do?"),
        ("user", "I run a bakery"),
        ("otto", "Tell me more."),
    ]


@pytest.mark.asyncio
async def test_many_reads_of_one_turn_at_once_ask_the_model_at_most_twice(
    reads: AsyncMock,
):
    async def slow(*args, **kwargs) -> DetailsRead:
        await asyncio.sleep(0.05)
        return _read(InterviewDetails(business="Bakery"))

    reads.side_effect = slow
    session = _session("I run a bakery")

    await asyncio.gather(
        *(interview_details.read_interview_details(session) for _ in range(25))
    )

    assert reads.await_count <= interview_details.MAX_READS_PER_TURN


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", ["vendor down", "no details"])
async def test_an_empty_read_is_kept_and_the_turn_is_not_read_again(
    reads: AsyncMock, empty: str
):
    if empty == "vendor down":
        reads.side_effect = RuntimeError("vendor down")
    else:
        reads.return_value = _read(None)
    session = _session("I run a bakery")

    for _ in range(5):
        assert await interview_details.read_interview_details(session) is None

    reads.assert_awaited_once()


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
async def test_an_answer_of_several_lines_is_read_whole(reads: AsyncMock):
    answer = "I run a bakery.\nWe open at six.\nSam does the ovens."

    await interview_details.read_interview_details(
        _session(answer, "Orders come in by email")
    )

    users = [said for who, said in reads.await_args.kwargs["lines"] if who == "user"]
    assert users == [answer, "Orders come in by email"]


@pytest.mark.asyncio
async def test_each_model_call_is_on_the_platform_cost_log(
    reads: AsyncMock, cost_log: AsyncMock
):
    session = _session("I run a bakery")

    await interview_details.read_interview_details(session)

    cost_log.assert_awaited_once()
    logged = cost_log.await_args.kwargs
    assert (logged["prompt_tokens"], logged["completion_tokens"]) == (800, 90)
    assert logged["user_id"] == "user-1"
    assert logged["charge_plan"] is False
    assert logged["cost_usd"] > 0
    assert logged["extra_metadata"]["session_id"] == session.session_id


@pytest.mark.asyncio
async def test_a_read_that_fails_beside_one_that_lands_leaves_the_kept_details(
    reads: AsyncMock,
):
    """Two reads of one turn run at once; the one that fails finishes last."""
    calls = 0

    async def one_lands_one_fails(*args, **kwargs) -> DetailsRead:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _read(InterviewDetails(business="Bakery"))
        await asyncio.sleep(0.05)
        raise RuntimeError("vendor down")

    reads.side_effect = one_lands_one_fails
    session = _session("I run a bakery")

    await asyncio.gather(
        interview_details.read_interview_details(session),
        interview_details.read_interview_details(session),
    )

    assert await interview_details.read_interview_details(session) == InterviewDetails(
        business="Bakery"
    )
