"""How one run moves a scheduled chat's failure streak.

Pure, so the rules are testable on their own: the caller loads the streak,
asks what this run changes, and carries it out (``service``).

* Three runs in a row that failed outright the same way pause the schedule,
  and the owner is told why, with the fix.
* Three runs in a row with the same failing step tell the owner once, and the
  schedule keeps running: the rest of what it does still works.
* A run that worked clears the streak. One that called no tool continues an
  open connection failure, and otherwise changes nothing.
* A neutral run (cancelled, our outage) changes nothing.
* The caller lets a run on a paused streak through only once the chat's jobs
  fire again: it was resumed some other way. Unless the run is neutral, the
  pause is over and the run counts from the start.
"""

from pydantic import BaseModel

from backend.copilot.scheduled_chats.outcomes import RunFailure, RunOutcome

PAUSE_AFTER = 3
TELL_AFTER = 3

_IDLE_CONTINUES = ("connection", "chat_link")


class Streak(BaseModel):
    failure: RunFailure
    # Runs in a row that failed this way.
    failures: int = 1
    # The latest of those runs that failed outright, in a row.
    whole_run_failures: int = 0
    notified: bool = False
    paused: bool = False


class StreakStep(BaseModel):
    # What to keep; None clears the streak.
    streak: Streak | None
    # Pause the schedule and tell the owner why.
    pause: bool = False
    # Tell the owner, and keep running.
    tell: bool = False
    # A streak the owner was told about has ended: withdraw what it raised.
    # Its pause alert is resolved; the notes stay, as a record.
    ended_notified: bool = False


def advance(streak: Streak | None, outcome: RunOutcome) -> StreakStep:
    if outcome.neutral:
        return StreakStep(streak=streak)
    failure, whole_run = outcome.failure, outcome.whole_run
    if failure is None:
        if not outcome.idle:
            return StreakStep(streak=None, ended_notified=_was_told(streak))
        if streak is None:
            return StreakStep(streak=None)
        if not _idle_continues(streak):
            if streak.paused:
                # Its pause is over, and a run that called no tool starts
                # nothing of its own.
                return StreakStep(streak=None, ended_notified=_was_told(streak))
            return StreakStep(streak=streak)
        failure, whole_run = streak.failure, True
    ended_notified = False
    if streak is not None and not streak.paused and streak.failure.key == failure.key:
        streak = streak.model_copy(
            update={
                "failures": streak.failures + 1,
                "whole_run_failures": (
                    streak.whole_run_failures + 1 if whole_run else 0
                ),
            }
        )
    else:
        # A different failure starts its own count. A run that fired while its
        # streak said paused was resumed some other way; it starts over too.
        ended_notified = _was_told(streak)
        streak = Streak(failure=failure, whole_run_failures=int(whole_run))
    if streak.whole_run_failures >= PAUSE_AFTER:
        return StreakStep(
            streak=streak.model_copy(update={"paused": True, "notified": True}),
            pause=True,
            ended_notified=ended_notified,
        )
    if streak.failures >= TELL_AFTER and not streak.notified:
        return StreakStep(
            streak=streak.model_copy(update={"notified": True}),
            tell=True,
            ended_notified=ended_notified,
        )
    return StreakStep(streak=streak, ended_notified=ended_notified)


def _idle_continues(streak: Streak) -> bool:
    return streak.failure.kind in _IDLE_CONTINUES


def _was_told(streak: Streak | None) -> bool:
    return streak is not None and streak.notified
