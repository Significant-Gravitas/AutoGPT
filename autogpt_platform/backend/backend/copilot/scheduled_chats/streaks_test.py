"""The three cases SECRT-2879 was filed for, run by run, plus the edges."""

from backend.copilot.scheduled_chats.outcomes import RunFailure, RunOutcome
from backend.copilot.scheduled_chats.streaks import Streak, advance

_DISCORD = RunFailure(
    key="connection:discord", kind="connection", provider="discord", tool="run_block"
)
_TWITTER = RunFailure(
    key="connection:twitter", kind="connection", provider="twitter", tool="run_block"
)
_BUDGET = RunFailure(key="turn:max_budget_exhausted", kind="budget")


def _outright(failure: RunFailure) -> RunOutcome:
    return RunOutcome(failure=failure, whole_run=True)


def _one_step(failure: RunFailure) -> RunOutcome:
    return RunOutcome(failure=failure, whole_run=False)


_IDLE = RunOutcome(idle=True)
_FINE = RunOutcome()
_NEUTRAL = RunOutcome(neutral=True)


def _drive(outcomes: list[RunOutcome], streak: Streak | None = None):
    steps = []
    for outcome in outcomes:
        step = advance(streak, outcome)
        steps.append(step)
        streak = step.streak
    return steps


def test_the_every_minute_monitor_pauses_on_its_third_run():
    """Run one could not reach Discord; the next two only repeated that from
    history, calling no tool. Those count, so the third run pauses."""
    steps = _drive([_outright(_DISCORD), _IDLE, _IDLE])

    assert [s.pause for s in steps] == [False, False, True]
    final = steps[-1].streak
    assert final is not None and final.paused and final.notified
    assert final.failures == 3


def test_a_monitor_that_cannot_post_to_discord_pauses_the_same_way():
    """An unlinked chat platform is as fixable as a missing connection, and a
    run that only repeats it from history counts the same."""
    unlinked = RunFailure(key="chat_link:discord", kind="chat_link", provider="discord")
    steps = _drive([_outright(unlinked), _IDLE, _IDLE])

    assert [s.pause for s in steps] == [False, False, True]


def test_the_twitter_briefing_is_told_once_and_keeps_running():
    """Twitter refused every morning, the briefing still went out: tell the
    owner on the third, never pause, never tell again."""
    steps = _drive([_one_step(_TWITTER)] * 5)

    assert [s.tell for s in steps] == [False, False, True, False, False]
    assert not any(s.pause for s in steps)
    final = steps[-1].streak
    assert final is not None and final.failures == 5 and not final.paused


def test_a_run_over_its_spending_limit_three_times_pauses():
    steps = _drive([_outright(_BUDGET)] * 3)

    assert steps[-1].pause


def test_one_good_run_in_between_starts_the_count_over():
    steps = _drive(
        [_outright(_DISCORD), _outright(_DISCORD), _FINE, _outright(_DISCORD)]
    )

    assert not any(s.pause for s in steps)
    final = steps[-1].streak
    assert final is not None and final.failures == 1


def test_a_different_failure_starts_its_own_count():
    steps = _drive([_outright(_DISCORD), _outright(_DISCORD), _outright(_TWITTER)])

    assert not steps[-1].pause
    final = steps[-1].streak
    assert final is not None
    assert final.failure.key == "connection:twitter"
    assert final.failures == 1


def test_a_step_failure_breaks_a_run_of_outright_failures():
    steps = _drive(
        [
            _outright(_DISCORD),
            _outright(_DISCORD),
            _one_step(_DISCORD),
            _outright(_DISCORD),
        ]
    )

    assert not any(s.pause for s in steps)
    final = steps[-1].streak
    assert final is not None and final.whole_run_failures == 1
    # Four runs with the same failing step: the owner was told on the third.
    assert steps[2].tell


def test_an_idle_run_without_an_open_connection_failure_changes_nothing():
    open_budget = Streak(failure=_BUDGET, failures=2, whole_run_failures=2)

    assert advance(None, _IDLE).streak is None
    assert advance(open_budget, _IDLE).streak == open_budget


def test_the_platforms_weather_changes_nothing():
    open_streak = Streak(failure=_DISCORD, failures=2, whole_run_failures=2)

    step = advance(open_streak, _NEUTRAL)

    assert step.streak == open_streak
    assert not step.pause and not step.tell


def test_ending_a_streak_the_owner_was_told_about_says_so():
    told = Streak(failure=_TWITTER, failures=3, notified=True)

    assert advance(told, _FINE).ended_notified
    assert not advance(Streak(failure=_TWITTER, failures=2), _FINE).ended_notified


def test_a_run_after_a_resume_the_streak_did_not_see_starts_over():
    """Resumed some other way (the chat's own schedule tool): the next run
    fires while the row still says paused, and is counted afresh."""
    paused = Streak(
        failure=_DISCORD, failures=3, whole_run_failures=3, notified=True, paused=True
    )

    step = advance(paused, _outright(_DISCORD))

    assert step.streak is not None
    assert step.streak.failures == 1 and not step.streak.paused
    assert step.ended_notified


def test_an_idle_run_after_a_resume_the_streak_did_not_see_starts_over():
    """Resumed some other way and still not connected, the monitor only
    repeats that from history. Its pause is over, so it is counted afresh,
    and the third such run pauses it again."""
    paused = Streak(
        failure=_DISCORD, failures=3, whole_run_failures=3, notified=True, paused=True
    )

    steps = _drive([_IDLE, _IDLE, _IDLE], paused)

    assert steps[0].ended_notified
    first = steps[0].streak
    assert first is not None and first.failures == 1 and not first.paused
    assert [s.pause for s in steps] == [False, False, True]


def test_an_idle_run_after_a_resume_ends_a_streak_it_does_not_continue():
    paused = Streak(
        failure=_BUDGET, failures=3, whole_run_failures=3, notified=True, paused=True
    )

    step = advance(paused, _IDLE)

    assert step.streak is None and step.ended_notified
