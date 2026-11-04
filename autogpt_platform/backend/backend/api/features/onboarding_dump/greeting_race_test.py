"""Which model's greeting the user gets: Sonnet when it answers, the fast
backup when Sonnet fails or is slow, and never a wait on both."""

import asyncio

import pytest

from backend.api.features.onboarding_dump.greeting_race import Attempt, IntroReply, race
from backend.api.features.onboarding_dump.models import SuggestedPrompt

SONNET = Attempt(model="sonnet", timeout=45)
HAIKU = Attempt(model="haiku", timeout=25)
FULL = IntroReply(
    greeting="Orders eat your mornings.",
    prompts=[SuggestedPrompt(title="Chase orders", prompt=".")],
)
HAIKU_FULL = IntroReply(
    greeting="Haiku heard you.",
    prompts=[SuggestedPrompt(title="Sort orders", prompt=".")],
)
SHORT = IntroReply(greeting="A greeting with too few tasks under it.")
EMPTY = IntroReply()
# Far past anything these tests wait for: reaching it means the race waited
# for the hedge when it should not have.
NEVER = 60.0


class Models:
    """A fake ``generate``: each model answers with its scripted reply, after
    ``delay`` seconds, or hangs until it is cancelled."""

    def __init__(
        self,
        replies: dict[str, IntroReply],
        hang: frozenset[str] = frozenset(),
        delay: dict[str, float] | None = None,
    ):
        self.replies = replies
        self.hang = hang
        self.delay = delay or {}
        self.called: list[str] = []
        self.cancelled: list[str] = []

    async def __call__(self, model: str, timeout: float) -> IntroReply:
        self.called.append(model)
        try:
            if model in self.hang:
                await asyncio.Event().wait()
            if model in self.delay:
                await asyncio.sleep(self.delay[model])
        except asyncio.CancelledError:
            self.cancelled.append(model)
            raise
        return self.replies[model]


async def _race(models: Models, hedge_after: float) -> IntroReply:
    # Correct code finishes at once; the bound only stops a broken race
    # from hanging the run.
    return await asyncio.wait_for(
        race(SONNET, HAIKU, models, hedge_after=hedge_after), timeout=10
    )


@pytest.mark.asyncio
async def test_sonnet_s_answer_is_used_and_the_backup_never_runs():
    models = Models({"sonnet": FULL})

    reply = await _race(models, hedge_after=NEVER)

    assert reply == FULL
    assert models.called == ["sonnet"]


@pytest.mark.asyncio
async def test_a_failed_sonnet_call_starts_the_backup_at_once():
    # Without waiting for the hedge: a deploy where Sonnet's call fails still
    # gets a greeting written from the take, well inside the plan step's wait.
    models = Models({"sonnet": EMPTY, "haiku": HAIKU_FULL})

    reply = await _race(models, hedge_after=NEVER)

    assert reply == HAIKU_FULL
    assert models.called == ["sonnet", "haiku"]


@pytest.mark.asyncio
async def test_a_slow_sonnet_is_raced_by_the_backup_and_cancelled():
    models = Models({"haiku": HAIKU_FULL}, hang=frozenset({"sonnet"}))

    reply = await _race(models, hedge_after=0.01)

    assert reply == HAIKU_FULL
    assert models.cancelled == ["sonnet"]


@pytest.mark.asyncio
async def test_sonnet_still_wins_when_it_answers_before_the_backup():
    models = Models(
        {"sonnet": FULL},
        hang=frozenset({"haiku"}),
        delay={"sonnet": 0.05},
    )

    reply = await _race(models, hedge_after=0.01)

    assert reply == FULL
    assert models.called == ["sonnet", "haiku"]
    assert models.cancelled == ["haiku"]


@pytest.mark.asyncio
async def test_a_full_backup_answer_beats_a_sonnet_one_with_too_few_tasks():
    models = Models({"sonnet": SHORT, "haiku": HAIKU_FULL})

    reply = await _race(models, hedge_after=NEVER)

    assert reply == HAIKU_FULL


@pytest.mark.asyncio
async def test_with_no_full_answer_the_first_greeting_is_kept():
    models = Models({"sonnet": SHORT, "haiku": EMPTY})

    reply = await _race(models, hedge_after=NEVER)

    assert reply == SHORT


@pytest.mark.asyncio
async def test_with_no_answer_at_all_the_reply_is_empty():
    models = Models({"sonnet": EMPTY, "haiku": EMPTY})

    reply = await _race(models, hedge_after=NEVER)

    assert reply.greeting is None
    assert not reply.full
