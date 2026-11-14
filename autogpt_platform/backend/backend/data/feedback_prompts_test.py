"""Tests for ``data.feedback_prompts`` against the real Postgres.

The limits and the claim's atomicity are the whole point of the table, so they
are proven against the database rather than a mocked query.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from prisma.enums import SubscriptionTier
from prisma.models import ChatSession, FeedbackPrompt, User

from backend.data import feedback_prompts
from backend.data.feedback_prompts import (
    NPS_AFTER,
    NPS_ROLLOUT_AT,
    FeedbackPromptKind,
    claim_feedback_prompt,
    count_feedback_prompts,
    get_feedback_prompt,
    update_feedback_prompt,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

T0 = datetime(2026, 12, 1, 12, tzinfo=UTC)
_BARRIER_TIMEOUT_SECONDS = 2


@pytest.fixture
async def user_id(server) -> AsyncIterator[str]:
    async for created in _users():
        yield created


@pytest.fixture
async def other_user_id(server) -> AsyncIterator[str]:
    async for created in _users():
        yield created


async def _users() -> AsyncIterator[str]:
    created = str(uuid4())
    await User.prisma().create(data={"id": created, "email": f"{created}@example.com"})
    yield created
    await User.prisma().delete_many(where={"id": created})


async def _make_nps_ready(user_id: str, signup: datetime, active_at: datetime) -> None:
    """A paying account that got something done at *active_at*."""
    await User.prisma().update(
        where={"id": user_id},
        data={"createdAt": signup, "subscriptionTier": SubscriptionTier.PRO},
    )
    await ChatSession.prisma().create(
        data={"userId": user_id, "updatedAt": active_at, "totalCompletionTokens": 10}
    )


async def _not_now(user_id: str, prompt_id: str | None) -> None:
    assert prompt_id is not None
    assert await update_feedback_prompt(
        user_id=user_id,
        prompt_id=prompt_id,
        reason=None,
        note=None,
        score=None,
        dismissed=True,
    )


async def test_shared_kinds_wait_a_week_for_each_other(user_id: str) -> None:
    assert await claim_feedback_prompt(user_id=user_id, kind="work_check", now=T0)
    for kind in ("work_check", "top_up"):
        assert (
            await claim_feedback_prompt(
                user_id=user_id, kind=kind, now=T0 + timedelta(days=6)
            )
            is None
        )
    assert await claim_feedback_prompt(
        user_id=user_id, kind="top_up", now=T0 + timedelta(days=7)
    )


async def test_top_up_waits_30_days(user_id: str) -> None:
    assert await claim_feedback_prompt(user_id=user_id, kind="top_up", now=T0)
    assert (
        await claim_feedback_prompt(
            user_id=user_id, kind="top_up", now=T0 + timedelta(days=29)
        )
        is None
    )
    assert await claim_feedback_prompt(
        user_id=user_id, kind="top_up", now=T0 + timedelta(days=30)
    )


async def test_two_not_nows_in_a_row_keep_quiet_for_30_days(user_id: str) -> None:
    await _not_now(
        user_id, await claim_feedback_prompt(user_id=user_id, kind="work_check", now=T0)
    )
    second = T0 + timedelta(days=7)
    await _not_now(
        user_id,
        await claim_feedback_prompt(user_id=user_id, kind="work_check", now=second),
    )

    assert (
        await claim_feedback_prompt(
            user_id=user_id, kind="work_check", now=second + timedelta(days=29)
        )
        is None
    )
    assert await claim_feedback_prompt(
        user_id=user_id, kind="work_check", now=second + timedelta(days=30)
    )


async def test_an_answer_between_not_nows_keeps_the_weekly_wait(user_id: str) -> None:
    await _not_now(
        user_id, await claim_feedback_prompt(user_id=user_id, kind="work_check", now=T0)
    )
    answered = await claim_feedback_prompt(
        user_id=user_id, kind="work_check", now=T0 + timedelta(days=7)
    )
    assert answered is not None
    await update_feedback_prompt(
        user_id=user_id,
        prompt_id=answered,
        reason=None,
        note=None,
        score=2,
        dismissed=False,
    )

    assert await claim_feedback_prompt(
        user_id=user_id, kind="work_check", now=T0 + timedelta(days=14)
    )


async def test_one_users_week_does_not_limit_another(
    user_id: str, other_user_id: str
) -> None:
    assert await claim_feedback_prompt(user_id=user_id, kind="work_check", now=T0)
    assert await claim_feedback_prompt(user_id=other_user_id, kind="work_check", now=T0)


async def test_nps_once_ever(user_id: str) -> None:
    due = NPS_ROLLOUT_AT + NPS_AFTER
    await _make_nps_ready(user_id, NPS_ROLLOUT_AT, active_at=due)

    assert await claim_feedback_prompt(user_id=user_id, kind="nps", now=due)
    assert (
        await claim_feedback_prompt(
            user_id=user_id, kind="nps", now=due + timedelta(days=400)
        )
        is None
    )


async def test_nps_needs_14_days(user_id: str) -> None:
    due = NPS_ROLLOUT_AT + NPS_AFTER
    await _make_nps_ready(user_id, NPS_ROLLOUT_AT, active_at=due)

    too_early = due - timedelta(seconds=1)
    assert (
        await claim_feedback_prompt(user_id=user_id, kind="nps", now=too_early) is None
    )
    assert await claim_feedback_prompt(user_id=user_id, kind="nps", now=due)


async def test_nps_skips_accounts_past_day_14_before_rollout(user_id: str) -> None:
    now = NPS_ROLLOUT_AT + timedelta(days=30)
    await _make_nps_ready(user_id, NPS_ROLLOUT_AT - timedelta(days=15), active_at=now)

    assert await claim_feedback_prompt(user_id=user_id, kind="nps", now=now) is None


async def test_nps_reaches_accounts_turning_14_days_after_rollout(user_id: str) -> None:
    now = NPS_ROLLOUT_AT + timedelta(days=1)
    await _make_nps_ready(user_id, NPS_ROLLOUT_AT - timedelta(days=13), active_at=now)

    assert await claim_feedback_prompt(user_id=user_id, kind="nps", now=now)


@pytest.mark.parametrize("tier", [SubscriptionTier.NO_TIER, SubscriptionTier.TRIAL])
async def test_nps_skips_accounts_not_paying(
    user_id: str, tier: SubscriptionTier
) -> None:
    due = NPS_ROLLOUT_AT + NPS_AFTER
    await _make_nps_ready(user_id, NPS_ROLLOUT_AT, active_at=due)
    await User.prisma().update(where={"id": user_id}, data={"subscriptionTier": tier})

    assert await claim_feedback_prompt(user_id=user_id, kind="nps", now=due) is None


async def test_nps_waits_for_something_done_this_week(user_id: str) -> None:
    due = NPS_ROLLOUT_AT + NPS_AFTER
    await _make_nps_ready(user_id, NPS_ROLLOUT_AT, active_at=due)
    later = due + timedelta(days=8)

    assert await claim_feedback_prompt(user_id=user_id, kind="nps", now=later) is None


async def test_nps_waits_for_the_shared_week(user_id: str) -> None:
    due = NPS_ROLLOUT_AT + NPS_AFTER
    await _make_nps_ready(user_id, NPS_ROLLOUT_AT, active_at=due)
    assert await claim_feedback_prompt(user_id=user_id, kind="top_up", now=due)

    assert await claim_feedback_prompt(user_id=user_id, kind="nps", now=due) is None


async def test_otto_mistake_once_per_session(user_id: str) -> None:
    assert await claim_feedback_prompt(
        user_id=user_id, kind="otto_mistake", session_id="chat-a", now=T0
    )
    assert (
        await claim_feedback_prompt(
            user_id=user_id, kind="otto_mistake", session_id="chat-a", now=T0
        )
        is None
    )
    assert await claim_feedback_prompt(
        user_id=user_id, kind="otto_mistake", session_id="chat-b", now=T0
    )


async def test_otto_mistake_without_a_chat_is_refused(user_id: str) -> None:
    assert await claim_feedback_prompt(user_id=user_id, kind="otto_mistake") is None
    assert await count_feedback_prompts(user_id=user_id, kind="otto_mistake") == 0


@pytest.mark.parametrize("kind", ["trial_cancel", "upgrade_checkout"])
async def test_unlimited_kinds_always_claim(
    user_id: str, kind: FeedbackPromptKind
) -> None:
    first = await claim_feedback_prompt(user_id=user_id, kind=kind, now=T0)
    second = await claim_feedback_prompt(user_id=user_id, kind=kind, now=T0)

    assert first and second and first != second
    assert await count_feedback_prompts(user_id=user_id, kind=kind) == 2


async def test_concurrent_claims_one_wins(user_id: str, mocker) -> None:
    """Both tasks get past the check before either inserts, unless the lock
    holds the second back; then the first's wait times out instead. Two
    different shared kinds take the same lock."""
    barrier = asyncio.Barrier(2)
    real_check = feedback_prompts._is_allowed

    async def check_then_wait_for_the_other(*args, **kwargs) -> bool:
        allowed = await real_check(*args, **kwargs)
        with suppress(TimeoutError, asyncio.BrokenBarrierError):
            await asyncio.wait_for(barrier.wait(), _BARRIER_TIMEOUT_SECONDS)
        return allowed

    mocker.patch.object(
        feedback_prompts, "_is_allowed", new=check_then_wait_for_the_other
    )

    claims = await asyncio.gather(
        claim_feedback_prompt(user_id=user_id, kind="work_check", now=T0),
        claim_feedback_prompt(user_id=user_id, kind="top_up", now=T0),
    )

    assert sum(claim is not None for claim in claims) == 1
    shown = await FeedbackPrompt.prisma().count(where={"userId": user_id})
    assert shown == 1


async def _claim_and_save_reason(user_id: str) -> str:
    prompt_id = await claim_feedback_prompt(user_id=user_id, kind="trial_cancel")
    assert prompt_id is not None
    saved = await update_feedback_prompt(
        user_id=user_id,
        prompt_id=prompt_id,
        reason="too_expensive",
        note=None,
        score=None,
        dismissed=False,
    )
    assert saved is not None
    return prompt_id


async def test_update_keeps_other_fields(user_id: str) -> None:
    prompt_id = await _claim_and_save_reason(user_id)

    updated = await update_feedback_prompt(
        user_id=user_id,
        prompt_id=prompt_id,
        reason=None,
        note="Too dear for now",
        score=None,
        dismissed=False,
    )

    assert updated is not None
    assert updated.reason == "too_expensive"
    assert updated.note == "Too dear for now"
    assert updated.answeredAt is not None
    assert updated.dismissedAt is None


async def test_dismissing_alone_does_not_answer(user_id: str) -> None:
    prompt_id = await claim_feedback_prompt(user_id=user_id, kind="trial_cancel")
    assert prompt_id is not None

    updated = await update_feedback_prompt(
        user_id=user_id,
        prompt_id=prompt_id,
        reason=None,
        note=None,
        score=None,
        dismissed=True,
    )

    assert updated is not None
    assert updated.dismissedAt is not None
    assert updated.answeredAt is None


async def test_note_is_stripped_of_postgres_control_characters(user_id: str) -> None:
    prompt_id = await _claim_and_save_reason(user_id)

    updated = await update_feedback_prompt(
        user_id=user_id,
        prompt_id=prompt_id,
        reason=None,
        note="null\x00byte",
        score=None,
        dismissed=False,
    )

    assert updated is not None
    assert updated.note == "nullbyte"


async def test_update_refuses_other_users_row(user_id: str, other_user_id: str) -> None:
    prompt_id = await _claim_and_save_reason(user_id)

    assert (
        await update_feedback_prompt(
            user_id=other_user_id,
            prompt_id=prompt_id,
            reason="other",
            note=None,
            score=None,
            dismissed=True,
        )
        is None
    )
    assert await get_feedback_prompt(user_id=other_user_id, prompt_id=prompt_id) is None
    row = await FeedbackPrompt.prisma().find_unique(where={"id": prompt_id})
    assert row is not None
    assert row.reason == "too_expensive"
    assert row.dismissedAt is None
