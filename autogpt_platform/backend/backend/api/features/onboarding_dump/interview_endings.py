"""How an onboarding interview ends: the arms of the ending experiment.

Every ending is an arm of a PostHog multivariate flag, one flag per mode
(``Experiment.ONBOARDING_INTERVIEW_ENDING_TEXT`` for the typed interview,
``..._VOICE`` for the voice call), so each arm is offered only in the mode it
fits and a person is enrolled only in the experiment for the mode they use.
The arm is read through ``backend.data.experiments``, which stores it on
first read: it stays the same whether or not the browser may report
analytics, and the ``analytics.*`` views split by it.

The arms are almost entirely frontend. The backend's part is when the answers
reach the onboarding pipeline: on the person's confirm for the arms that end
on one (:data:`CONFIRM_ENDINGS`), as soon as the interview closes for every
other arm and for anyone not enrolled.
"""

import logging
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.data import experiments
from backend.data.experiments import Experiment
from backend.data.redis_client import get_redis_async
from backend.util.feature_flag import Flag, evaluate_feature_flag

logger = logging.getLogger(__name__)

InterviewMode = Literal["text", "voice"]


class InterviewEnding(str, Enum):
    """The endings; each value is the variant key of the PostHog flag."""

    PROFILE_CARD = "02-profile-card"
    SUMMARY_CHIPS = "03-summary-chips"
    TEAM_REVEAL = "04-team-reveal"
    COVERAGE_RING = "06-coverage-ring"
    TRANSCRIPT_HIGHLIGHTS = "08-transcript-highlights"
    FIRST_TASK = "09-first-task"


ENDING_EXPERIMENTS: dict[InterviewMode, Experiment] = {
    "text": Experiment.ONBOARDING_INTERVIEW_ENDING_TEXT,
    "voice": Experiment.ONBOARDING_INTERVIEW_ENDING_VOICE,
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
            InterviewEnding.PROFILE_CARD,
            InterviewEnding.SUMMARY_CHIPS,
            InterviewEnding.TEAM_REVEAL,
            InterviewEnding.TRANSCRIPT_HIGHLIGHTS,
            InterviewEnding.FIRST_TASK,
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


# --- A tester's pick ---------------------------------------------------------
#
# With the picker flag on, a tester picks the ending on the step, and the
# start request carries the pick. A pick that fits the mode wins over the
# experiment's arm, and is never an enrolment: the experiment is not read, so
# no assignment is stored and no exposure is sent.

# The typed interview's arm is stamped when it closes, so its pick is kept
# until then, one key per person.
_TEXT_PICK_KEY = "onboarding:interview:ending_pick:{user_id}"
_TEXT_PICK_TTL_S = 7 * 24 * 3600


async def picker_enabled(user_id: str) -> bool:
    """Whether *user_id* may pick their interview's ending; False when the
    flag cannot be read."""
    try:
        enabled, _ = await evaluate_feature_flag(
            Flag.ONBOARDING_INTERVIEW_ENDING_PICKER, user_id, default=False
        )
    except Exception:
        logger.warning("Ending picker flag could not be read", exc_info=True)
        return False
    return bool(enabled)


def fits_mode(pick: str | None, mode: InterviewMode) -> bool:
    return pick in {ending.value for ending in MODE_ENDINGS[mode]}


async def honoured_pick(
    user_id: str, mode: InterviewMode, pick: str | None
) -> str | None:
    """*pick*, when the picker is on for *user_id* and the arm fits *mode*;
    None otherwise, and the pick is ignored."""
    if pick is None or not fits_mode(pick, mode):
        return None
    return pick if await picker_enabled(user_id) else None


async def keep_text_pick(user_id: str, pick: str | None) -> None:
    """Keep a tester's pick for their typed interview until it closes. A
    start without one, from a tester, forgets any pick kept before ("Default"
    in the picker); anyone else's start leaves nothing behind."""
    if not await picker_enabled(user_id):
        return
    key = _TEXT_PICK_KEY.format(user_id=user_id)
    try:
        redis = await get_redis_async()
        if pick is not None and fits_mode(pick, "text"):
            await redis.set(key, pick, ex=_TEXT_PICK_TTL_S)
        else:
            await redis.delete(key)
    except Exception:
        logger.warning("Ending pick not kept", exc_info=True)


class TextPick(BaseModel):
    """What a tester's pick says of how their typed interview closes.

    ``governs`` is True when a pick decides the arm rather than the
    experiment: a pick kept from the start, or a tester's pick that cannot
    be read. ``arm`` is the pick to stamp, or None to stamp none: a pick
    that cannot be read, or one kept while the picker has since been turned
    off. The experiment is read only when no pick governs.
    """

    model_config = ConfigDict(frozen=True)

    arm: str | None = None
    governs: bool = False


async def text_pick(user_id: str) -> TextPick:
    """The tester's pick their typed interview closes on (:class:`TextPick`)."""
    try:
        redis = await get_redis_async()
        pick = await redis.get(_TEXT_PICK_KEY.format(user_id=user_id))
    except Exception as exc:
        logger.warning(f"Ending pick unreadable: {type(exc).__name__}")
        return TextPick(governs=await picker_enabled(user_id))
    if isinstance(pick, bytes):
        pick = pick.decode()
    if pick is None:
        return TextPick()
    if not fits_mode(pick, "text") or not await picker_enabled(user_id):
        return TextPick(governs=True)
    return TextPick(arm=pick, governs=True)
