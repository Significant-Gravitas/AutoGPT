"""How an onboarding interview ends: the arms of the ending experiment.

Every ending is an arm of a PostHog multivariate flag, one flag per mode
(``Experiment.ONBOARDING_INTERVIEW_ENDING_TEXT`` for the typed interview), so
each arm is offered only in the mode it fits and a person is enrolled only in
the experiment for the mode they use. The arm is read through
``backend.data.experiments``, which stores it on first read: it stays the same
whether or not the browser may report analytics, and the ``analytics.*`` views
split by it.

The arms are almost entirely frontend. The backend's part is when the answers
reach the onboarding pipeline: on the person's confirm for the arms that end
on one (:data:`CONFIRM_ENDINGS`), as soon as the interview closes for every
other arm and for anyone not enrolled.
"""

import logging
from enum import Enum
from typing import Literal

from backend.data import experiments
from backend.data.experiments import Experiment

logger = logging.getLogger(__name__)

InterviewMode = Literal["text", "voice"]


class InterviewEnding(str, Enum):
    """The ten endings; each value is the variant key of the PostHog flag."""

    SPOKEN_YES = "01-spoken-yes"
    PROFILE_CARD = "02-profile-card"
    SUMMARY_CHIPS = "03-summary-chips"
    TEAM_REVEAL = "04-team-reveal"
    SOFT_COUNTDOWN = "05-soft-countdown"
    COVERAGE_RING = "06-coverage-ring"
    HANG_UP = "07-hang-up"
    TRANSCRIPT_HIGHLIGHTS = "08-transcript-highlights"
    FIRST_TASK = "09-first-task"
    BACKGROUND_HAND_OFF = "10-background-hand-off"


ENDING_EXPERIMENTS: dict[InterviewMode, Experiment] = {
    "text": Experiment.ONBOARDING_INTERVIEW_ENDING_TEXT,
}

# The arms each mode's flag offers: the ones that fit that mode.
MODE_ENDINGS: dict[InterviewMode, frozenset[InterviewEnding]] = {
    "text": frozenset(
        {
            InterviewEnding.PROFILE_CARD,
            InterviewEnding.SUMMARY_CHIPS,
            InterviewEnding.TEAM_REVEAL,
            InterviewEnding.COVERAGE_RING,
            InterviewEnding.TRANSCRIPT_HIGHLIGHTS,
            InterviewEnding.FIRST_TASK,
        }
    ),
    "voice": frozenset(
        {
            InterviewEnding.SPOKEN_YES,
            InterviewEnding.PROFILE_CARD,
            InterviewEnding.TEAM_REVEAL,
            InterviewEnding.SOFT_COUNTDOWN,
            InterviewEnding.COVERAGE_RING,
            InterviewEnding.HANG_UP,
            InterviewEnding.FIRST_TASK,
            InterviewEnding.BACKGROUND_HAND_OFF,
        }
    ),
}

# The arms that end on a screen the person confirms: their answers are
# handed off on that confirm, with any corrections made there.
CONFIRM_ENDINGS: frozenset[str] = frozenset(
    {
        InterviewEnding.PROFILE_CARD.value,
        InterviewEnding.SUMMARY_CHIPS.value,
        InterviewEnding.TRANSCRIPT_HIGHLIGHTS.value,
        InterviewEnding.FIRST_TASK.value,
    }
)


def hands_off_on_close(ending: str | None) -> bool:
    """Whether an interview ending on *ending* is handed off as it closes.

    True for every arm but the confirm arms, and for anyone not enrolled
    (``None``) or on a variant this code does not know.
    """
    return ending not in CONFIRM_ENDINGS


async def assigned_ending(user_id: str, mode: InterviewMode) -> str | None:
    """*user_id*'s arm of the ending experiment for *mode*, read server-side.

    ``None`` while they are not enrolled, and when the arm cannot be read: an
    experiment never holds up an interview, which then ends as an unenrolled
    one does.
    """
    try:
        assignment = await experiments.assign(user_id, ENDING_EXPERIMENTS[mode])
    except Exception:
        logger.warning(
            f"Onboarding interview ending not assigned for user {user_id}",
            exc_info=True,
        )
        return None
    if assignment is None:
        return None
    if assignment.variant not in {ending.value for ending in MODE_ENDINGS[mode]}:
        logger.warning(
            f"Onboarding interview ending {assignment.variant!r} does not fit "
            f"the {mode} mode"
        )
    return assignment.variant
