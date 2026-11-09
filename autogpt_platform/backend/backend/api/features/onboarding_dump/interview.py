"""The onboarding interview on the web side: who may hold one, how it closes,
and where its answers go.

Starting an interview, admitting each of its turns and charging it to its
budget happen here, for every route that touches one
(:func:`start_onboarding_interview`, :func:`admit_onboarding_interview_turn`,
:func:`enforce_onboarding_interview_allowance`). The rules for which turns an
interview takes are in ``backend.copilot.onboarding_interview``.

An interview closes in one place, :func:`close_onboarding_interview`,
whatever ends it: the reply cap, the scope guard, the end signal, or the
person on screen (:func:`finalize_onboarding_interview`, the confirm). Every
dispatched turn settles once it ends (:func:`close_after_turn`), so a turn
that ended the interview is closed here whichever way it ended it. The
person's arm of the ending experiment (``interview_endings``) decides when
the answers are handed off: as it closes, or on the confirm.

An interview ends where a brain dump does: with what the user said. So it is
handed over as a typed take, and everything downstream runs exactly as it does
for a dump: the quality gate; extraction into ``upsert_business_understanding``,
with the words themselves in ``additional_notes`` and from there in Otto's
``<user_context>``; the greeting and suggested prompts; the provider and team
recommendations; and the ``/status``, ``/intro`` and recommendation reads the
loading screen already polls.

Two layers, so where the words come from can change without the pipeline
noticing. :func:`hand_off_interview_transcript` takes the user's words from
any interview; the typed interview's come from its chat session, a voice
call's from its transcript of record (:func:`close_voice_interview`).

A person has one interview, typed or spoken. The voice call start route
refuses a call once the typed interview is over; the typed interview is
refused while a call of theirs is live, and is over once a call's words
became the take (:func:`refuse_while_a_call_holds_the_interview`). A call
whose hand-off another take beat is ``superseded``, and so is a typed
interview closed after a call took the interview.
"""

import asyncio
import logging
import re
from collections.abc import Sequence

from fastapi import BackgroundTasks, HTTPException
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from pydantic import BaseModel, ConfigDict

from backend.api.features.onboarding_dump import db as dump_db
from backend.api.features.onboarding_dump import quality, service
from backend.api.features.onboarding_dump.interview_endings import (
    InterviewMode,
    assigned_ending,
    hands_off_on_close,
)
from backend.api.features.onboarding_dump.models import (
    MAX_CORRECTION_CHARS,
    MAX_TYPED_CHARS,
    FinalizeResponse,
)
from backend.api.features.onboarding_dump.rate_limit import (
    enforce_interview_turn_budget,
)
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.expert_kickoff import is_hidden_chat_message
from backend.copilot.model import (
    INTERVIEW_STAGE_CLOSED,
    INTERVIEW_STAGE_HANDED_OFF,
    INTERVIEW_STAGE_WRAP_UP_ISSUED,
    ChatMessage,
    ChatSession,
    ChatSessionInfo,
    get_chat_session,
    get_chat_session_metadata,
    update_session_interview_ending,
    update_session_interview_stage,
)
from backend.copilot.onboarding_interview import (
    INTERVIEW_CLOSED,
    InterviewEndReason,
    InterviewTurnPlan,
    InterviewTurnRefused,
    check_onboarding_interview_turn,
    count_answered_turns,
    find_onboarding_interview_session,
    get_or_create_onboarding_interview_session,
    interview_is_over,
    interview_reached_cap,
    is_onboarding_interview,
    kick_off_onboarding_interview,
    plan_interview_turn,
    user_has_spoken,
)
from backend.copilot.onboarding_interview_scope import (
    interview_scope_classifier,
    judge_interview_scope,
)
from backend.copilot.pending_message_helpers import (
    StreamRegistryUnavailable,
    is_turn_in_flight,
)
from backend.copilot.service import (
    strip_injected_context_for_display,
    strip_server_injected_tags,
)
from backend.data.db_accessors import chat_db
from backend.data.onboarding import get_onboarding_voice_call_id
from backend.data.redis_client import get_redis_async
from backend.util.background import spawn_background_task
from backend.util.exceptions import NotFoundError
from backend.voice_relay import locks

logger = logging.getLogger(__name__)

INTERVIEW_RECORDING_PREFIX = "interview-"

# The 409 detail for an interview with a turn already running in it, worded
# as the chat continuation route words it. The start route answers it with
# the interview to follow, since that turn is usually Otto's greeting.
TURN_IN_PROGRESS = "turn_in_progress"

# The take a hand-off answers with when a newer take owns the user's row
# (``service.finalize_typed_dump``).
SUPERSEDED = "superseded"

# Failed takes that are still the pipeline's last word on an interview's
# words: a verdict on them, or a newer take that must not be replaced.
_FINAL_FAILURES = frozenset(
    {quality.NO_USABLE_SPEECH, quality.INSUFFICIENT_CONTENT, SUPERSEDED}
)

# How long a close waits for the turn that ends the interview to finish
# before leaving the interview to the next request that touches it.
GOODBYE_WAIT_SECONDS = 600.0
_GOODBYE_POLL_SECONDS = 2.0

# The chat's answer to an ``ask_question`` card (``QuestionDock`` and
# ``buildAnswersMessage`` in the frontend): Otto's questions quoted, the
# user's answers under them, the platform's own words around both.
_ANSWERS_HEADER = "**Here are my answers:**"
_ANSWERS_FOOTER_RE = re.compile(r"\n*Please proceed\b[^\n]*\s*$")
_LIST_MARKER_RE = re.compile(r"^[-*+]\s+")


async def enforce_onboarding_interview_allowance(user_id: str) -> None:
    """Whether *user_id* may hold the next turn of an onboarding interview.

    The one decision for the kickoff and for every turn after it, so nobody
    is greeted and then refused. The interview runs before the plan step, so
    it needs no plan and draws on no plan's allowance (the executor lets an
    interview's turns past its own allowance gate, ``budget_gate.refusal_for``,
    the same way). It is bounded instead by:

    - its own turn budget, on the personalization budget's rules and on
      every tier (``rate_limit.enforce_interview_turn_budget``), charged here
      once a turn is about to be dispatched;
    - one interview per user (its session id, recorded on the user's
      onboarding row);
    - the code-owned reply cap (``INTERVIEW_TURN_CAP``);
    - no turn once it is closed or handed off (409 here, a refused turn on
      the executor);
    - the scope guard (``onboarding_interview_scope``).

    Raises:
        HTTPException: 429 with the turn budget spent, 503 while it cannot be
            read.
    """
    await enforce_interview_turn_budget(user_id)


class StartedInterview(BaseModel):
    model_config = ConfigDict(frozen=True)

    session: ChatSession
    # Otto's first turn, when this call started it; None when the interview
    # was already under way and there was nothing to start.
    turn_id: str | None


async def start_onboarding_interview(user_id: str) -> StartedInterview:
    """Open the user's onboarding interview and have Otto speak first.

    The one call a route makes to start or reopen an interview. The user has
    one interview, so a second call adopts it; an interview the user has
    already spoken in, or whose greeting was written, is reopened with
    nothing new started. So is one that is over, so the person sees how it
    ended and the way on; one over but not yet handed off is handed off in
    the background, as the chat routes do. The greeting is charged to the
    interview's budget only once nothing else is running in it. Otto's first
    turn, once this call dispatches it, is settled when it ends, as every
    dispatched turn is (:func:`record_onboarding_interview_turn_started`).

    Raises:
        HTTPException: 409 when a turn is already running in it (the client
            follows that turn's stream instead), 429 at the cap on turns
            running at once, whatever
            :func:`refuse_while_a_call_holds_the_interview` raises while a
            voice call holds the interview, and whatever
            :func:`enforce_onboarding_interview_allowance` raises.
    """
    await refuse_while_a_call_holds_the_interview(user_id)
    session = await get_or_create_onboarding_interview_session(user_id)
    if interview_is_over(session):
        if session.metadata.onboarding_interview_stage != INTERVIEW_STAGE_HANDED_OFF:
            spawn_background_task(
                hand_off_closed_interview(user_id, session.session_id),
                name=f"interview-hand-off-{session.session_id[:12]}",
            )
        return StartedInterview(session=session, turn_id=None)
    if user_has_spoken(session) or count_answered_turns(session.messages) > 0:
        return StartedInterview(session=session, turn_id=None)
    await _refuse_if_busy(session.session_id)
    await enforce_onboarding_interview_allowance(user_id)
    try:
        turn_id = await kick_off_onboarding_interview(session)
    except ConcurrentTurnLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except InterviewTurnRefused:
        # The user spoke between the read above and the kickoff: reopen.
        return StartedInterview(session=session, turn_id=None)
    if turn_id is None:
        latest = await get_chat_session(session.session_id, user_id)
        if latest is not None and count_answered_turns(latest.messages) > 0:
            # The greeting was written after the read above.
            return StartedInterview(session=latest, turn_id=None)
        # Another request took the session between the busy check and here,
        # most often the first click of a double click.
        raise HTTPException(status_code=409, detail=TURN_IN_PROGRESS)
    await record_onboarding_interview_turn_started(
        user_id, session.session_id, plan_interview_turn(session)
    )
    return StartedInterview(session=session, turn_id=turn_id)


async def _refuse_if_busy(session_id: str) -> None:
    try:
        busy = await is_turn_in_flight(session_id)
    except StreamRegistryUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="Chat service degraded, retry shortly",
            headers={"Retry-After": "30"},
        ) from exc
    if busy:
        raise HTTPException(status_code=409, detail=TURN_IN_PROGRESS)


async def refuse_while_a_call_holds_the_interview(user_id: str) -> None:
    """Refuse the typed interview while a voice call holds the person's
    interview: a call is live, or a call's words became the interview's take.

    Raises:
        HTTPException: 409 ``call_in_progress`` naming the live call, which
            the person can rejoin or end; 409 ``onboarding_interview_closed``
            naming the call that took the interview, whose ending the step
            shows; 503 while neither can be read.
    """
    try:
        live = await locks.live_call_of(await get_redis_async(), user_id)
        taken_by = None if live else await call_that_took_the_interview(user_id)
    except Exception as exc:
        logger.warning(
            f"Onboarding interview: voice calls unreadable: {type(exc).__name__}"
        )
        raise _voice_calls_unreadable() from exc
    if live:
        raise HTTPException(
            status_code=409,
            detail={"code": locks.CALL_IN_PROGRESS, "call_id": live},
        )
    if taken_by:
        raise HTTPException(
            status_code=409, detail={"code": INTERVIEW_CLOSED, "call_id": taken_by}
        )


async def call_that_took_the_interview(user_id: str) -> str | None:
    """The person's voice call whose words became their interview's take, or
    None.

    Only their latest call can be (``UserOnboarding.voiceCallId``), once its
    hand-off was accepted: its record says so for a day, and after that the
    dump row does, while it still holds that call's take.
    """
    call_id = await get_onboarding_voice_call_id(user_id)
    if call_id is None:
        return None
    record = await locks.read_call(await get_redis_async(), call_id)
    if record is not None:
        return call_id if record.took_the_interview else None
    dump = await dump_db.get_dump(user_id)
    if (
        dump is not None
        and dump.recordingId == interview_recording_id(call_id)
        and dump.status != BrainDumpStatus.failed
    ):
        return call_id
    return None


def _voice_calls_unreadable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Onboarding interview temporarily unavailable, retry shortly",
        headers={"Retry-After": "30"},
    )


async def admit_onboarding_interview_turn(
    session: ChatSessionInfo,
    user_id: str,
    *,
    is_user_message: bool = True,
    expert_kickoff: bool = False,
    has_attachments: bool = False,
) -> InterviewTurnPlan:
    """Admit a message into an onboarding interview, as a new turn or queued
    on the running one.

    Every route that takes a message into an interview calls this in place of
    its own paywall and allowance checks. ``session`` is the route's own
    read of the session row, so the stage it holds is the database's. The
    interview's budget is charged later, by the route, once the message is
    about to start a turn (:func:`enforce_onboarding_interview_allowance`): a
    message that rides a running turn, or a retransmit of one already taken,
    starts nothing and costs nothing.

    Raises:
        HTTPException: 409 once the interview is over, 422 for a request an
            interview never takes (a non-user turn, an expert kickoff, files),
            and whatever :func:`refuse_while_a_call_holds_the_interview`
            raises while a voice call holds the interview.
    """
    full = await get_chat_session(session.session_id, user_id)
    if full is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # The cached copy can trail a stage recorded a moment ago.
    full = full.model_copy(update={"metadata": session.metadata})
    try:
        plan = check_onboarding_interview_turn(
            full,
            is_user_message=is_user_message,
            expert_kickoff=expert_kickoff,
            has_attachments=has_attachments,
        )
    except InterviewTurnRefused as refused:
        if refused.code != INTERVIEW_CLOSED:
            raise HTTPException(status_code=422, detail=refused.code) from refused
        if session.metadata.onboarding_interview_stage != INTERVIEW_STAGE_HANDED_OFF:
            # An interview that is over closes and hands off on its own; this
            # catches one whose close or hand-off never ran.
            spawn_background_task(
                hand_off_closed_interview(user_id, session.session_id),
                name=f"interview-hand-off-{session.session_id[:12]}",
            )
        raise HTTPException(status_code=409, detail=refused.code) from refused
    await refuse_while_a_call_holds_the_interview(user_id)
    return plan


async def record_onboarding_interview_turn_started(
    user_id: str, session_id: str, plan: InterviewTurnPlan
) -> None:
    """Record what a dispatched interview turn is for, and settle the
    interview once that turn is over, in one background task per turn
    (:func:`_settle_turn`). Called once the turn is dispatched or queued,
    never for a message that rides a running turn."""
    if plan.wraps_up:
        try:
            await update_session_interview_stage(
                session_id, user_id, INTERVIEW_STAGE_WRAP_UP_ISSUED
            )
        except Exception:
            logger.warning(
                f"Onboarding interview {session_id[:12]}: wrap-up not recorded",
                exc_info=True,
            )
    spawn_background_task(
        _settle_turn(user_id, session_id, plan),
        name=f"interview-settle-{session_id[:12]}",
    )


async def _settle_turn(user_id: str, session_id: str, plan: InterviewTurnPlan) -> None:
    """The background task of one dispatched interview turn: close the
    interview once the turn ends, if the turn ended it.

    The goodbye at the reply cap ends it. Any other turn is judged on scope
    beside it while the scope guard is on, and one it finds out of scope ends
    it too. Whatever the server decides, Otto's end signal may have closed
    the interview during the turn; :func:`close_after_turn` finishes that
    close as well.
    """
    reason: InterviewEndReason | None = "cap" if plan.says_goodbye else None
    if reason is None and interview_scope_classifier() is not None:
        reason = await _scope_verdict(user_id, session_id)
    await close_after_turn(user_id, session_id, reason)


async def _scope_verdict(user_id: str, session_id: str) -> InterviewEndReason | None:
    """Why the scope guard ends the interview on the turn just dispatched, or
    None when it lets the turn pass or cannot judge it."""
    try:
        session = await get_chat_session(session_id, user_id)
        if session is None:
            return None
        return await judge_interview_scope(session)
    except Exception:
        logger.warning(
            f"Onboarding interview {session_id[:12]}: scope not judged",
            exc_info=True,
        )
        return None


async def close_after_turn(
    user_id: str, session_id: str, reason: InterviewEndReason | None = None
) -> None:
    """Wait for the interview's running turn to end, then close the interview
    if that turn ended it.

    *reason* is the server's own, known when the turn was dispatched: the
    goodbye at the reply cap, or the scope guard's verdict on the turn. A
    goodbye at the cap that failed before it was written leaves the interview
    open, so the user can send again. Whatever the reason, a turn in which
    Otto's end signal closed the interview is closed here as confirmed
    (``reason="confirmed"``): the end signal only moves the stage, so this is
    where its ending arm is stamped, and an arm that ends on a confirm keeps
    its ending screen. An interview already handed off is left alone.

    A wait that runs out, or a restart that ends it, leaves the close to the
    next request on the interview (:func:`admit_onboarding_interview_turn`).
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + GOODBYE_WAIT_SECONDS
    try:
        while True:
            await asyncio.sleep(_GOODBYE_POLL_SECONDS)
            if not await is_turn_in_flight(session_id):
                break
            if loop.time() >= deadline:
                logger.info(
                    f"Onboarding interview {session_id[:12]}: closing turn still "
                    "running, close left to the next request"
                )
                return
        session, stored = await _read_interview(user_id, session_id)
        stage = stored.metadata.onboarding_interview_stage
        if stage == INTERVIEW_STAGE_HANDED_OFF:
            return
        if reason == "cap" and not interview_reached_cap(session):
            reason = None
        if reason is None and stage == INTERVIEW_STAGE_CLOSED:
            reason = "confirmed"
        if reason is not None:
            await close_onboarding_interview(user_id, session_id, reason=reason)
    except Exception:
        logger.warning(
            f"Onboarding interview {session_id[:12]}: close after its turn failed",
            exc_info=True,
        )


async def hand_off_closed_interview(user_id: str, session_id: str) -> None:
    """The next request's fallback for an interview that is over and not yet
    handed off; any other interview is left alone.

    One that used up its replies but never closed is closed now, as the cap
    closes it, and its arm decides the hand-off. One closed with its arm
    stamped is handed off whatever the arm: that covers someone who left an
    ending screen without confirming, and a hand-off that failed. One closed
    with no arm stamped is closed here as its turn's settle closes it
    (:func:`close_after_turn`), so the arm is read and decides the hand-off:
    Otto's end signal closes an interview inside a turn, and a request can
    arrive before that turn's settle has stamped the arm, or after a restart
    lost it. A confirm arm then keeps its ending screen, and anyone not
    enrolled is handed off now.

    Background work: its pipeline jobs run here, rather than after a response.
    """
    try:
        session, stored = await _read_interview(user_id, session_id)
    except NotFoundError:
        return
    stage = stored.metadata.onboarding_interview_stage
    if stage == INTERVIEW_STAGE_HANDED_OFF:
        return
    if stage == INTERVIEW_STAGE_CLOSED:
        if stored.metadata.onboarding_interview_ending is None:
            await close_onboarding_interview(user_id, session_id, reason="confirmed")
        else:
            await finalize_onboarding_interview(user_id, session_id, None)
    elif interview_reached_cap(session):
        await close_onboarding_interview(user_id, session_id, reason="cap")


class InterviewClosure(BaseModel):
    """How a close left an interview."""

    model_config = ConfigDict(frozen=True)

    # The person's arm of the ending experiment; None for anyone not
    # enrolled.
    ending: str | None
    # The take the answers became, once handed off; None while the arm waits
    # for the person's confirm.
    hand_off: FinalizeResponse | None


async def close_onboarding_interview(
    user_id: str,
    session_id: str,
    *,
    reason: InterviewEndReason,
    confirmed: bool = False,
    corrections: str | None = None,
    background_tasks: BackgroundTasks | None = None,
) -> InterviewClosure:
    """Close an onboarding interview: the one place every interview closes.

    Marks the stage ``closed`` first, forward-only, so every later turn is
    refused while the rest runs. Then reads the person's arm of the text
    ending experiment on the server (never from the client), stamps it onto
    the session once, and starts the hand-off the arm calls for: now for an
    automatic arm or anyone not enrolled, and for an arm that ends on a
    confirm only when this call is that confirm (*confirmed*). A confirm's
    *corrections* reach the hand-off as the person's own words.

    Safe to call again: an interview closes and is handed off once, and a
    call on one already handed off describes its take as it stands. It is
    handed off once the pipeline has ruled on its words
    (:func:`hand_off_is_final`); a hand-off the pipeline never ruled on
    leaves it closed but not handed off, for the next call to retry. Closing
    also drops any question Otto left waiting, so Home never lists it.
    *background_tasks* runs the pipeline's jobs after a route's response;
    without it they run before this returns.

    A voice call whose words already became the interview's take keeps it:
    the typed interview is handed off as ``superseded``, its words unused.

    Raises:
        NotFoundError: no onboarding interview of *user_id*'s has this id.
        HTTPException: 503 while the person's voice calls cannot be read.
    """
    session, stored = await _read_interview(user_id, session_id)
    ending = stored.metadata.onboarding_interview_ending
    if stored.metadata.onboarding_interview_stage == INTERVIEW_STAGE_HANDED_OFF:
        return InterviewClosure(
            ending=ending, hand_off=await _handed_off_take(user_id, session_id)
        )
    if await update_session_interview_stage(
        session_id, user_id, INTERVIEW_STAGE_CLOSED
    ):
        logger.info(f"Onboarding interview {session_id[:12]} closed: {reason}")
    if stored.metadata.pending_question is not None:
        await _drop_pending_question(user_id, session_id)
    try:
        taken_by = await call_that_took_the_interview(user_id)
    except Exception as exc:
        logger.warning(
            f"Onboarding interview {session_id[:12]}: voice calls unreadable: "
            f"{type(exc).__name__}"
        )
        raise _voice_calls_unreadable() from exc
    if taken_by is not None:
        logger.info(
            f"Onboarding interview {session_id[:12]}: voice call "
            f"{taken_by[:12]} took the interview"
        )
        await update_session_interview_stage(
            session_id, user_id, INTERVIEW_STAGE_HANDED_OFF
        )
        return InterviewClosure(ending=ending, hand_off=_superseded())
    if ending is None:
        ending = await _stamp_ending(user_id, session_id, "text")
    if not (confirmed or hands_off_on_close(ending)):
        return InterviewClosure(ending=ending, hand_off=None)
    tasks = background_tasks if background_tasks is not None else BackgroundTasks()
    response = await hand_off_interview_transcript(
        user_id,
        session_id,
        with_corrections(interview_transcript(session.messages), corrections),
        tasks,
    )
    if hand_off_is_final(response):
        await update_session_interview_stage(
            session_id, user_id, INTERVIEW_STAGE_HANDED_OFF
        )
    if background_tasks is None:
        await tasks()
    return InterviewClosure(ending=ending, hand_off=response)


def hand_off_is_final(response: FinalizeResponse) -> bool:
    """Whether the pipeline's answer to a hand-off is final for the
    interview's words, so the interview is handed off.

    It is unless the take failed without a ruling on the words: a failure
    with a verdict on them (``no_usable_speech``, ``insufficient_content``)
    is final, and so is ``superseded``, since a newer take owns the row and
    this one must never replace it. A failure the pipeline never ruled on
    (the judge could not be asked, or no code at all) is not; nor is a
    refusal it raised (429, 503), which never reaches here.
    """
    if response.status != BrainDumpStatus.failed:
        return True
    return response.error_code in _FINAL_FAILURES


def voice_hand_off_closes_interview(response: FinalizeResponse) -> bool:
    """Whether a voice call's hand-off closes the person's interview.

    It does for an accepted take, and for ``superseded``: another take owns
    the interview and this one must never replace it. Unlike the typed
    interview's, a verdict on the words leaves it open, since a new call is
    new words; so does a hand-off the pipeline never ruled on, which the
    call's finish route hands off again.
    """
    return (
        response.status != BrainDumpStatus.failed or response.error_code == SUPERSEDED
    )


def _superseded() -> FinalizeResponse:
    return FinalizeResponse(
        status=BrainDumpStatus.failed,
        input_mode=BrainDumpInputMode.typed,
        error_code=SUPERSEDED,
    )


def _nothing_said() -> FinalizeResponse:
    """The quality gate's own "nothing usable" answer, which the onboarding
    screens already handle, for an interview where nothing was said."""
    return FinalizeResponse(
        status=BrainDumpStatus.failed,
        input_mode=BrainDumpInputMode.typed,
        error_code=quality.NO_USABLE_SPEECH,
    )


async def close_voice_interview(
    user_id: str,
    call_id: str,
    transcript: str,
    background_tasks: BackgroundTasks,
    *,
    reason: InterviewEndReason,
    confirmed: bool = False,
    corrections: str | None = None,
    first_task: str | None = None,
) -> InterviewClosure:
    """The voice interview's hand-off: what the voice relay calls once a
    call's transcript of record is made, and what the call's finish route
    calls for a confirm or a retry.

    *transcript* is the person's side of the call's transcript of record, one
    turn per line, and *call_id* names the take it becomes: each call is a
    take of its own, so a later call replaces an earlier one's take as any
    re-take does. The person's arm of the voice ending experiment, read here,
    decides when: an automatic arm, or no enrolment, hands the words to the
    onboarding pipeline now; an arm that ends on a confirm leaves that to the
    confirm, which calls this again with *confirmed* and any *corrections*
    made on its screen. Until then the call's record keeps the transcript.
    *first_task* is the task the call's end tool carried; the hand-off goes on
    the person's own words, as the typed interview's does, and the call's
    record keeps the task.

    A call whose interview another take already owns is answered
    ``superseded`` and its words are left out: the person called again since,
    or their typed interview is over. A call where nothing was said is
    answered ``no_usable_speech`` whatever the arm, as there is nothing to
    confirm. Whether the answer closes the interview is
    :func:`voice_hand_off_closes_interview`.
    """
    if await _call_superseded(user_id, call_id):
        logger.info(f"Voice onboarding interview {call_id[:12]} superseded: {reason}")
        return InterviewClosure(ending=None, hand_off=_superseded())
    words = with_corrections(transcript, corrections)
    if not words.strip():
        return InterviewClosure(ending=None, hand_off=_nothing_said())
    ending = await assigned_ending(user_id, "voice")
    if not (confirmed or hands_off_on_close(ending)):
        return InterviewClosure(ending=ending, hand_off=None)
    logger.info(f"Voice onboarding interview {call_id[:12]} handed off: {reason}")
    response = await hand_off_interview_transcript(
        user_id, call_id, words, background_tasks
    )
    return InterviewClosure(ending=ending, hand_off=response)


async def _call_superseded(user_id: str, call_id: str) -> bool:
    """Whether another take owns the interview *call_id* would hand off into:
    a later call of the person's, or their typed interview, once it is over."""
    latest = await get_onboarding_voice_call_id(user_id)
    if latest is not None and latest != call_id:
        return True
    typed = await find_onboarding_interview_session(user_id)
    return typed is not None and interview_is_over(typed)


async def _drop_pending_question(user_id: str, session_id: str) -> None:
    try:
        await chat_db().clear_session_pending_question(session_id, user_id)
    except Exception:
        logger.warning(
            f"Onboarding interview {session_id[:12]}: waiting question not cleared",
            exc_info=True,
        )


async def finalize_onboarding_interview(
    user_id: str,
    session_id: str,
    background_tasks: BackgroundTasks | None,
    *,
    corrections: str | None = None,
) -> FinalizeResponse:
    """The person's confirm: close the interview, if it is still open, and
    hand it to the brain-dump pipeline now, whatever the arm.

    What an ending screen's confirm calls, with any corrections the person
    made on it, and what a route calls when the person ends the interview on
    screen. Runs through :func:`close_onboarding_interview`, so it is
    idempotent the same way: an interview already handed off is not handed
    off again, so a newer take the user made since is never replaced by it,
    and the response describes the take as it stands. Only the session's
    owner can finalize it, and a session of any other kind is treated as
    missing.
    """
    closure = await close_onboarding_interview(
        user_id,
        session_id,
        reason="user",
        confirmed=True,
        corrections=corrections,
        background_tasks=background_tasks,
    )
    if closure.hand_off is None:
        raise RuntimeError(f"Onboarding interview {session_id} was not handed off")
    return closure.hand_off


async def _read_interview(
    user_id: str, session_id: str
) -> tuple[ChatSession, ChatSessionInfo]:
    """The interview with its rows, and the database's read of its metadata,
    which a cached copy can trail; :class:`NotFoundError` unless it is one of
    *user_id*'s onboarding interviews."""
    session = await get_chat_session(session_id, user_id)
    stored = await get_chat_session_metadata(session_id, user_id)
    if session is None or stored is None or not is_onboarding_interview(stored):
        raise NotFoundError(f"Onboarding interview {session_id} not found")
    return session.model_copy(update={"metadata": stored.metadata}), stored


async def _stamp_ending(
    user_id: str, session_id: str, mode: InterviewMode
) -> str | None:
    """Read the person's arm and stamp it onto the interview; an arm that
    cannot be stamped still decides the hand-off."""
    ending = await assigned_ending(user_id, mode)
    if ending is None:
        return None
    try:
        await update_session_interview_ending(session_id, user_id, ending)
    except Exception:
        logger.warning(
            f"Onboarding interview {session_id[:12]}: ending arm not stamped",
            exc_info=True,
        )
    return ending


def with_corrections(transcript: str, corrections: str | None) -> str:
    """The words to hand off: the interview's, then what the person changed
    on an ending screen, as their own words too.

    The corrections are their latest word, so they are kept within the typed
    take's limit and the interview's words are cut to make room before them.
    """
    extra = strip_server_injected_tags(corrections or "").strip()
    extra = extra[:MAX_CORRECTION_CHARS]
    if not extra:
        return transcript
    if not transcript.strip():
        return extra
    room = MAX_TYPED_CHARS - len(extra) - 1
    return f"{transcript[:room]}\n{extra}"


def interview_recording_id(interview_id: str) -> str:
    """The take an interview becomes. One per interview, so handing the same
    interview over twice is the typed path's idempotent repeat, and a later
    interview or recording replaces it like any other re-take."""
    return f"{INTERVIEW_RECORDING_PREFIX}{interview_id}"


def interview_transcript(messages: Sequence[ChatMessage]) -> str:
    """The user's own words from an interview, one message per line.

    Otto's side is left out, and so are rows the user never saw (the hidden
    kickoff), rows sent from another session (``from_session_id``), and the
    blocks the server put in front of a message (``<user_context>``,
    ``<voice_turn>``, ...). An answer given on an
    ``ask_question`` card keeps only the answer: the stored row also quotes
    Otto's questions and the platform's framing (:func:`_own_answers`). The
    pipeline treats the whole text as something the user said: the greeting
    reflects it back and the quality gate judges it, so a word of Otto's in
    it would be put in their mouth. Cut to the typed take's limit.
    """
    lines: list[str] = []
    for message in messages:
        if (
            message.role != "user"
            or is_hidden_chat_message(message.metadata)
            or _sent_from_another_session(message.metadata)
        ):
            continue
        text = _own_answers(
            strip_injected_context_for_display(message.content or "").strip()
        )
        if text:
            lines.append(text)
    return "\n".join(lines)[:MAX_TYPED_CHARS]


def _sent_from_another_session(metadata: dict | None) -> bool:
    return bool((metadata or {}).get("from_session_id"))


def _own_answers(text: str) -> str:
    """*text* without the ``ask_question`` framing, when it is a card's answer.

    The card's message is ``**Here are my answers:**``, then for each
    question ``> {question}`` and the answer under it (one line per pick when
    several apply), then ``Please proceed.``. What is left is the answers,
    one per line, several picks joined with commas.
    """
    if not text.startswith(_ANSWERS_HEADER):
        return text
    body = _ANSWERS_FOOTER_RE.sub("", text[len(_ANSWERS_HEADER) :])
    answers: list[list[str]] = []
    current: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            if current:
                answers.append(current)
            current = []
        elif stripped:
            current.append(_LIST_MARKER_RE.sub("", stripped))
    if current:
        answers.append(current)
    return "\n".join(", ".join(answer) for answer in answers)


async def hand_off_interview_transcript(
    user_id: str,
    interview_id: str,
    transcript: str,
    background_tasks: BackgroundTasks,
) -> FinalizeResponse:
    """Run the brain-dump pipeline on an interview's transcript, as a typed take.

    An interview where nothing was said leaves the user's dump row alone, as
    the typed route does for empty text, and answers with the quality gate's
    own "nothing usable" code, which the onboarding screens already handle.
    """
    text = transcript.strip()[:MAX_TYPED_CHARS]
    if not text:
        return _nothing_said()
    return await service.finalize_typed_dump(
        user_id,
        interview_recording_id(interview_id),
        text,
        background_tasks,
    )


async def _handed_off_take(user_id: str, session_id: str) -> FinalizeResponse:
    dump = await dump_db.get_dump(user_id)
    if dump is None:
        # The only hand-off that leaves no row is one where nothing was said
        # (:func:`hand_off_interview_transcript`).
        return _nothing_said()
    if dump.recordingId != interview_recording_id(session_id):
        return _superseded()
    return FinalizeResponse(
        status=dump.status,
        input_mode=dump.inputMode,
        error_code=dump.errorCode,
    )
