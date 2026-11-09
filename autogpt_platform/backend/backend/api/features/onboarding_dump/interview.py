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
refused while a call of theirs is live, and is over once a call took the
interview (:func:`refuse_while_a_call_holds_the_interview`). A call takes it
when its words were accepted and the call ran to its close, or the person
confirmed it; a call that was cut off leaves the interview open, for a later
call or the typed interview to replace its take. The take the person made
last owns the interview: their first typed message after a call ended
supersedes the call's words, unless they took the interview first
(:func:`supersede_the_latest_call`). A call whose hand-off another take beat
is ``superseded``, and so is a typed interview closed after a call took the
interview.
"""

import asyncio
import logging
import re
import time
from collections.abc import Sequence
from typing import Literal

from fastapi import BackgroundTasks, HTTPException
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from pydantic import BaseModel, ConfigDict, ValidationError

from backend.api.features.onboarding_dump import db as dump_db
from backend.api.features.onboarding_dump import quality, service
from backend.api.features.onboarding_dump.interview_endings import (
    InterviewMode,
    assigned_ending,
    hands_off_on_close,
)
from backend.api.features.onboarding_dump.models import (
    INTERVIEW_RECORDING_PREFIX,
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
from backend.data.onboarding import (
    get_onboarding_voice_call_id,
    get_onboarding_voice_interview_call_id,
)
from backend.data.redis_client import AsyncRedisClient, get_redis_async
from backend.data.redis_helpers import as_str
from backend.util.background import spawn_background_task
from backend.util.exceptions import NotFoundError
from backend.voice_relay import locks

logger = logging.getLogger(__name__)

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
        live = await locks.live_call_holding(
            await get_redis_async(), user_id, time.time()
        )
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

    A call takes the interview when the pipeline accepted its words and the
    call ran to its close, or the person confirmed it; one that was cut off
    leaves the interview open (``voice_handoff``). The hand-off writes
    that on the person's onboarding row, which outlives the call's record,
    and the row is what is read.
    """
    return await get_onboarding_voice_interview_call_id(user_id)


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

    A message refused here or later, at the budget, supersedes nothing: the
    person's latest call is superseded once the message is dispatched or
    queued (:func:`typed_message_accepted`). One whose words took the
    interview first is refused here already.

    Raises:
        HTTPException: 409 once the interview is over, 422 for a request an
            interview never takes (a non-user turn, an expert kickoff, files),
            and whatever :func:`refuse_while_a_call_holds_the_interview` and
            :func:`refuse_once_the_latest_call_took_it` raise while a voice
            call holds the interview.
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
    await refuse_once_the_latest_call_took_it(user_id)
    return plan


async def refuse_once_the_latest_call_took_it(user_id: str) -> None:
    """Refuse a typed message once the words of the person's latest take, the
    latest call of theirs that has words (:func:`_calls_back_to_a_take`),
    took the interview on the call's record, which the hand-off claims a
    moment before it writes the onboarding row. Claims nothing.

    Raises:
        HTTPException: 409 ``onboarding_interview_closed`` naming the call;
            503 while the calls cannot be read.
    """
    try:
        calls = await _calls_back_to_a_take(await get_redis_async(), user_id)
    except Exception as exc:
        logger.warning(
            f"Onboarding interview: voice calls unreadable: {type(exc).__name__}"
        )
        raise _voice_calls_unreadable() from exc
    for record in calls:
        if record.interview == locks.INTERVIEW_TAKEN:
            raise HTTPException(
                status_code=409,
                detail={"code": INTERVIEW_CLOSED, "call_id": record.call_id},
            )


async def typed_message_accepted(user_id: str) -> None:
    """The person's typed message was dispatched or queued: it supersedes
    their latest call's words (:func:`supersede_the_latest_call`). The
    message is the person's already, so nothing here refuses it; should the
    call's words have taken the interview a moment before, the next turn's
    admission answers the 409."""
    try:
        await supersede_the_latest_call(user_id)
    except HTTPException as refusal:
        logger.info(
            f"Onboarding interview: a typed message after the latest call took "
            f"nothing from it ({refusal.status_code})"
        )


async def supersede_the_latest_call(user_id: str) -> None:
    """Make the person's typing the newer take than their voice calls, once
    the latest has ended: what they type now owns the interview, and a call's
    hand-off, should it land later, is answered ``superseded``.

    Claimed on the person's latest take (:func:`_calls_back_to_a_take`): a
    call that settled without words is no take, and the take before it is
    the one whose hand-off claims the interview. That hand-off claims on its
    own record too (:func:`locks.claim_interview`), so of the two exactly one
    wins. Then, unless the take's words won, on each call after it that
    settled without words. A call whose record expired has no hand-off left
    to land. Called once no call of the person's is live
    (:func:`refuse_while_a_call_holds_the_interview`), so the latest call has
    ended: by the typed finish, and for each typed message once it is
    dispatched or queued (:func:`typed_message_accepted`). The typed start
    never calls it, since a greeting the person has only looked at does not
    outrank a complete call.

    Raises:
        HTTPException: 409 ``onboarding_interview_closed`` naming the call
            whose words took the interview first; 503 while the calls cannot
            be read.
    """
    took: str | None = None
    try:
        redis = await get_redis_async()
        for record in reversed(await _calls_back_to_a_take(redis, user_id)):
            standing = await locks.claim_interview(
                redis, record.call_id, locks.INTERVIEW_SUPERSEDED
            )
            if standing == locks.INTERVIEW_TAKEN:
                took = record.call_id
                break
    except Exception as exc:
        logger.warning(
            f"Onboarding interview: voice calls unreadable: {type(exc).__name__}"
        )
        raise _voice_calls_unreadable() from exc
    if took is not None:
        raise HTTPException(
            status_code=409, detail={"code": INTERVIEW_CLOSED, "call_id": took}
        )


async def _calls_back_to_a_take(
    redis: AsyncRedisClient, user_id: str
) -> list[locks.CallRecord]:
    """The person's calls from the onboarding row's latest back along each
    record's ``prev``, through every one that settled without words
    (:func:`call_words`) or that takes no claim yet, starting or running
    (:func:`locks.claim_interview`), to the first that has words or is still
    to settle: the person's latest take, last. Stops at a record that
    expired or is not the person's."""
    calls: list[locks.CallRecord] = []
    current = await get_onboarding_voice_call_id(user_id)
    while current is not None and all(call.call_id != current for call in calls):
        record = await locks.read_call(redis, current)
        if record is None or record.uid != user_id:
            break
        calls.append(record)
        if not _takes_no_claim(record) and call_words(record) != "no_words":
            break
        current = record.prev
    return calls


def _takes_no_claim(record: locks.CallRecord) -> bool:
    """Whether the call is still starting or running, which no claim on the
    interview reaches."""
    return record.status is None or record.status in locks.RUNNING


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
    enrolled is handed off now. One whose hand-off is owed is handed off as
    owed, with the corrections its close was handed
    (:func:`hand_off_owed_typed_interview`).

    Background work: its pipeline jobs run here, rather than after a response.
    """
    try:
        owed = await _owed_hand_off(user_id)
    except Exception:
        logger.warning(
            f"Onboarding interview {session_id[:12]}: owed hand-off unreadable",
            exc_info=True,
        )
        owed = None
    if owed is not None and owed.session_id == session_id:
        await hand_off_owed_typed_interview(user_id)
        return
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
            # The server's own hand-off, not the person's confirm: an
            # interview a call took is answered superseded, never refused.
            await close_onboarding_interview(
                user_id, session_id, reason="user", confirmed=True
            )
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
    # A voice call's answer ``superseded`` because another take owns the
    # interview (:func:`_call_standing`), ruled before the dump row was
    # touched; not one the row gave.
    outranked: bool = False


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
    leaves it closed but not handed off, for the next call to retry. A
    ``superseded`` answer while the dump row still holds the take of a call
    the person's typing beat leaves it closed too, its hand-off owed and
    started once the row is free (:func:`_waits_on_a_beaten_call`). Closing
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
        return InterviewClosure(ending=ending, hand_off=superseded_take())
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
    if hand_off_is_final(response) and not await _waits_on_a_beaten_call(
        user_id, session_id, response, corrections
    ):
        await update_session_interview_stage(
            session_id, user_id, INTERVIEW_STAGE_HANDED_OFF
        )
    if background_tasks is None:
        await tasks()
    return InterviewClosure(ending=ending, hand_off=response)


async def _waits_on_a_beaten_call(
    user_id: str,
    session_id: str,
    response: FinalizeResponse,
    corrections: str | None,
) -> bool:
    """Whether a ``superseded`` answer to the typed interview's hand-off only
    means the dump row still holds the take of a call the person's typing
    beat (:func:`supersede_the_latest_call`), so the answer is not final:
    the interview stays closed, and its hand-off is owed, with *corrections*
    (:func:`hand_off_owed_typed_interview`).

    The owed hand-off starts once the row is free. The call's hand-off gives
    the row up and starts it (``voice_handoff``). A call whose hand-off has
    answered already gives nothing up any more, so its take is given up here
    while no pipeline has run it (:func:`service.give_up_unrun_take`), and
    the owed hand-off started. The hand-off is owed before the call's answer
    is read, so one side or the other always starts it; the REST API's sweep
    hands off any left behind. A read that fails counts as not final.
    """
    if response.error_code != SUPERSEDED:
        return False
    try:
        call_id = await _beaten_call_on_the_row(user_id)
        if call_id is None:
            return False
        await _owe_the_hand_off(user_id, session_id, corrections)
        record = await locks.read_call(await get_redis_async(), call_id)
        if record is not None and record.hand_off is None:
            logger.info(
                f"Onboarding interview {session_id[:12]}: a call's take still "
                f"holds the dump row; handed off once the call's hand-off gives "
                f"it up"
            )
            return True
        recording_id = interview_recording_id(call_id)
        await service.give_up_unrun_take(user_id, recording_id, SUPERSEDED)
        if await _take_still_runs(user_id, recording_id):
            logger.info(
                f"Onboarding interview {session_id[:12]}: a call's take still "
                f"runs on the dump row; handed off by the sweep once it is done"
            )
            return True
        start_owed_typed_hand_off(user_id)
    except Exception as exc:
        logger.warning(
            f"Onboarding interview {session_id[:12]}: voice calls unreadable: "
            f"{type(exc).__name__}"
        )
    return True


async def _beaten_call_on_the_row(user_id: str) -> str | None:
    """The call whose take is on the person's dump row, when a newer take
    beat it to the interview: its record, or the record of a call of the
    person's after it, says ``superseded`` (:func:`_superseded_from`). None
    for any other take."""
    row = await dump_db.get_dump(user_id)
    if row is None or not row.recordingId.startswith(INTERVIEW_RECORDING_PREFIX):
        return None
    call_id = row.recordingId[len(INTERVIEW_RECORDING_PREFIX) :]
    if not await _superseded_from(await get_redis_async(), user_id, call_id):
        return None
    return call_id


async def _superseded_from(redis: AsyncRedisClient, user_id: str, call_id: str) -> bool:
    """Whether *call_id*'s record, or that of a later call of the person's,
    says ``superseded``: the calls are walked from the onboarding row's
    latest back along each record's ``prev`` to *call_id*, as
    :func:`_call_standing` walks them. The person's typing claims on their
    latest take, so a claim on a call after *call_id* outranks it too."""
    seen: set[str] = set()
    current = await get_onboarding_voice_call_id(user_id)
    while current is not None and current not in seen:
        seen.add(current)
        record = await locks.read_call(redis, current)
        if record is None or record.uid != user_id:
            break
        if record.interview == locks.INTERVIEW_SUPERSEDED:
            return True
        if current == call_id:
            return False
        current = record.prev
    record = await locks.read_call(redis, call_id)
    return (
        record is not None
        and record.uid == user_id
        and record.interview == locks.INTERVIEW_SUPERSEDED
    )


async def _take_still_runs(user_id: str, recording_id: str) -> bool:
    """Whether the dump row still holds *recording_id*'s take where a newer
    take cannot replace it: waiting for the pipeline, or in it."""
    row = await dump_db.get_dump(user_id)
    return (
        row is not None
        and row.recordingId == recording_id
        and row.status in _TAKE_STILL_RUNS
    )


# The row statuses in which a take holds the dump row against a newer typed
# take (``db.start_dump``).
_TAKE_STILL_RUNS = frozenset(
    {
        BrainDumpStatus.recording_uploaded,
        BrainDumpStatus.transcribing,
        BrainDumpStatus.transcribed,
        BrainDumpStatus.extracting,
    }
)

# Typed interviews whose hand-off is owed (:func:`_waits_on_a_beaten_call`):
# one entry per person, naming their interview and the corrections its close
# was handed (:func:`owed_hand_off_key`), and an index of the people owed
# one, by when, for the REST API's sweep.
OWED_HAND_OFFS_KEY = "onboarding:interview:hand_off:owed"
# An owed hand-off still not handed off this long after it was first owed
# lapses; the person's next visit to the step hands the interview off.
OWED_HAND_OFF_TTL_S = 24 * 3600


def owed_hand_off_key(user_id: str) -> str:
    return f"{OWED_HAND_OFFS_KEY}:{user_id}"


class OwedHandOff(BaseModel):
    """A typed interview's hand-off a close made and the pipeline left owed."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    corrections: str | None = None
    # When it was first owed, in seconds since the epoch.
    at: float


async def _owed_hand_off(user_id: str) -> OwedHandOff | None:
    raw = as_str(await (await get_redis_async()).get(owed_hand_off_key(user_id)))
    if raw is None:
        return None
    try:
        return OwedHandOff.model_validate_json(raw)
    except ValidationError:
        logger.warning("Onboarding interview: an owed hand-off is unreadable")
        return None


async def _owe_the_hand_off(
    user_id: str, session_id: str, corrections: str | None
) -> None:
    """Record the person's typed hand-off as owed, with *corrections* as the
    hand-off keeps them, until a day after it was first owed."""
    earlier = await _owed_hand_off(user_id)
    now = time.time()
    owed = OwedHandOff(
        session_id=session_id,
        corrections=_own_corrections(corrections) or None,
        at=(
            earlier.at
            if earlier is not None and earlier.session_id == session_id
            else now
        ),
    )
    redis = await get_redis_async()
    await redis.set(
        owed_hand_off_key(user_id),
        owed.model_dump_json(),
        ex=max(1, int(owed.at + OWED_HAND_OFF_TTL_S - now)),
    )
    await redis.zadd(OWED_HAND_OFFS_KEY, {user_id: owed.at})


async def _forget_the_hand_off(user_id: str) -> None:
    redis = await get_redis_async()
    await redis.delete(owed_hand_off_key(user_id))
    await redis.zrem(OWED_HAND_OFFS_KEY, user_id)


def start_owed_typed_hand_off(user_id: str) -> asyncio.Task:
    """Start the person's owed typed hand-off in the background
    (:func:`hand_off_owed_typed_interview`)."""
    return spawn_background_task(
        hand_off_owed_typed_interview(user_id),
        name=f"interview-owed-hand-off-{user_id[:12]}",
    )


async def hand_off_owed_typed_interview(user_id: str) -> None:
    """Hand off the person's typed interview if its hand-off is owed: a close
    handed it off while the take of a call the person's typing beat held the
    dump row (:func:`_waits_on_a_beaten_call`). It is handed off as that
    close was, with the corrections it was handed, and forgotten once it is
    handed off or gone.

    What the call's hand-off starts once it gives up the row its take held
    (``voice_handoff``), what a typed request starts for a closed interview
    (:func:`hand_off_closed_interview`), and what the REST API's sweep runs
    for any left behind. An interview waiting for the person's confirm owes
    nothing, so this never hands one off ahead of it.

    Background work: its pipeline jobs run here, rather than after a response.
    """
    owed = await _owed_hand_off(user_id)
    if owed is None:
        # Lapsed, or already handed off.
        await _forget_the_hand_off(user_id)
        return
    try:
        _, stored = await _read_interview(user_id, owed.session_id)
    except NotFoundError:
        await _forget_the_hand_off(user_id)
        return
    if stored.metadata.onboarding_interview_stage != INTERVIEW_STAGE_CLOSED:
        await _forget_the_hand_off(user_id)
        return
    await close_onboarding_interview(
        user_id,
        owed.session_id,
        reason="user",
        confirmed=True,
        corrections=owed.corrections,
    )
    _, stored = await _read_interview(user_id, owed.session_id)
    if stored.metadata.onboarding_interview_stage == INTERVIEW_STAGE_HANDED_OFF:
        await _forget_the_hand_off(user_id)


async def hand_off_owed_typed_interviews(stop: asyncio.Event | None = None) -> int:
    """Hand off, one at a time, every typed interview whose hand-off is owed
    (:func:`hand_off_owed_typed_interview`); *stop* ends it between them. For
    the REST API's sweep. Returns how many were owed."""
    redis = await get_redis_async()
    # People whose owed hand-off has lapsed leave the index.
    await redis.zremrangebyscore(
        OWED_HAND_OFFS_KEY, "-inf", time.time() - OWED_HAND_OFF_TTL_S
    )
    members = await redis.zrange(OWED_HAND_OFFS_KEY, 0, -1)
    owed = [
        user_id
        for user_id in (
            as_str(member) for member in members if isinstance(member, (bytes, str))
        )
        if user_id
    ]
    for user_id in owed:
        if stop is not None and stop.is_set():
            break
        try:
            await hand_off_owed_typed_interview(user_id)
        except Exception as error:
            logger.warning(
                f"Onboarding interview: an owed hand-off failed: "
                f"{type(error).__name__}"
            )
    return len(owed)


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
    """Whether a voice call's hand-off is over for the call, which is then
    ``handed_off``.

    It is for an accepted take, which takes the person's interview when the
    call ran to its close or the person confirmed it
    (``voice_handoff``), and for ``superseded``: another take owns the
    interview and this one must never replace it. Unlike the typed
    interview's, a verdict on the words leaves the interview open, since a
    new call is new words; so does a hand-off the pipeline never ruled on,
    which the call's finish route hands off again.
    """
    return (
        response.status != BrainDumpStatus.failed or response.error_code == SUPERSEDED
    )


def superseded_take() -> FinalizeResponse:
    """The answer for a take another take beat: it never replaces the row."""
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
    ending: str | None = None,
    confirmed: bool = False,
    corrections: str | None = None,
    first_task: str | None = None,
    unanswered: bool = False,
) -> InterviewClosure:
    """The voice interview's hand-off: what the REST API runs once the relay
    has made a call's transcript of record (``voice_handoff``), and what the
    call's finish route runs for a confirm or a retry.

    *transcript* is the person's side of the call's transcript of record, one
    turn per line, and *call_id* names the take it becomes: each call is a
    take of its own, so a later call replaces an earlier one's take as any
    re-take does. The person's arm of the voice ending experiment, *ending*,
    decides when: an automatic arm, or no enrolment, hands the words to the
    onboarding pipeline now; an arm that ends on a confirm leaves that to the
    confirm, which calls this again with *confirmed* and any *corrections*
    made on its screen. Until then the call's record keeps the transcript.
    The arm is read by the call's start route, where flags are read, and
    comes with the call's record.
    *first_task* is the task the call's end tool carried; the hand-off goes on
    the person's own words, as the typed interview's does, and the call's
    record keeps the task.

    A call whose interview another take already owns is answered
    ``superseded`` and its words are left out: a later call of the person's
    has words of record, they typed into their interview after the call
    ended, or their typed interview is over (:func:`_call_standing`). While a
    later call of theirs could still become a take, this raises
    :class:`LaterCallUnsettled` and hands nothing off. A call where nothing
    was said is answered ``no_usable_speech`` whatever the arm, as there is
    nothing to confirm.
    Whether the answer is over for the call is
    :func:`voice_hand_off_closes_interview`; whether it takes the interview
    is the call's hand-off's to claim (``voice_handoff``).

    *unanswered* says no answer accepting the call's words is recorded on the
    call: none at all, or a failure the call's finish hands off again. A
    take the row holds ``transcribed`` that this call did not judge was then
    claimed by an earlier hand-off that stopped before it answered, and its
    words are judged again rather than read as a ruling.
    """
    standing = await _call_standing(user_id, call_id)
    if standing == "superseded":
        logger.info(f"Voice onboarding interview {call_id[:12]} superseded: {reason}")
        return InterviewClosure(ending=None, hand_off=superseded_take(), outranked=True)
    if standing == "waiting":
        raise LaterCallUnsettled
    if not transcript.strip():
        # Nothing was said, so nothing to confirm or correct: corrections
        # alone never make a call a take (:func:`call_words`).
        return InterviewClosure(ending=None, hand_off=_nothing_said())
    words = with_corrections(transcript, corrections)
    if not (confirmed or hands_off_on_close(ending)):
        return InterviewClosure(ending=ending, hand_off=None)
    logger.info(f"Voice onboarding interview {call_id[:12]} handed off: {reason}")
    response = await _hand_off_call_words(
        user_id, call_id, words, background_tasks, unanswered=unanswered
    )
    return InterviewClosure(ending=ending, hand_off=response)


async def _hand_off_call_words(
    user_id: str,
    call_id: str,
    words: str,
    background_tasks: BackgroundTasks,
    *,
    unanswered: bool,
) -> FinalizeResponse:
    text = words.strip()[:MAX_TYPED_CHARS]
    recording_id = interview_recording_id(call_id)
    take = await service.finalize_typed_take(
        user_id, recording_id, text, background_tasks, call_take=True
    )
    if (
        take.judged
        or not unanswered
        or take.response.status != BrainDumpStatus.transcribed
        or not await service.reopen_unjudged_take(user_id, recording_id)
    ):
        return take.response
    logger.info(
        f"Voice onboarding interview {call_id[:12]}: an earlier hand-off left its "
        f"words unjudged; judging them now"
    )
    take = await service.finalize_typed_take(
        user_id, recording_id, text, background_tasks, call_take=True
    )
    return take.response


class LaterCallUnsettled(Exception):
    """A later call of the person's may still become a take. The earlier
    call's hand-off waits for it, writing nothing."""


# Answers that settle a call without words of record: nothing usable was
# heard, or no transcript could be made. A verdict on words that were said,
# such as ``insufficient_content``, is words.
NO_WORDS = frozenset(
    {
        quality.NO_USABLE_SPEECH,
        locks.TRANSCRIPTION_UNAVAILABLE,
        locks.RECORDING_UNAVAILABLE,
    }
)

CallWords = Literal["words", "no_words", "unsettled"]


def call_words(record: locks.CallRecord) -> CallWords:
    """Whether a call has words of record, a take: a claim on the person's
    interview, a transcript with words in it, or an accepted hand-off. A call
    whose transcript is empty, or that recorded a no-words answer
    (:data:`NO_WORDS`), settled without words. Anything else is still to
    settle: running, ended, or being transcribed."""
    if record.interview is not None:
        return "words"
    if record.hand_off is not None and record.hand_off.error_code in NO_WORDS:
        return "no_words"
    if record.transcript_of_record is not None:
        return "words" if record.transcript_of_record.strip() else "no_words"
    return "words" if record.accepted else "unsettled"


CallStanding = Literal["superseded", "waiting", "clear"]


async def _call_standing(user_id: str, call_id: str) -> CallStanding:
    """Whether another take owns the interview *call_id* would hand off into.

    It is ``superseded`` once the person typed after the call ended (the
    call's record says so), once their typed interview is over, or once a
    later call of theirs has words of record (:func:`call_words`): the calls
    after this one are walked from the onboarding row's latest call back
    along each record's ``prev``, and the first that has words supersedes it.
    A later call that settled without words is passed over, as no take; one
    still to settle makes the hand-off wait (``waiting``), so no hand-off runs
    while a later call could still become a take. ``clear`` once the walk
    reaches this call, or the chain breaks: a later call with no record that
    holds no lock died before it began, and a record that is not the
    person's is no call of theirs.
    """
    redis = await get_redis_async()
    if await locks.interview_of(redis, call_id) == locks.INTERVIEW_SUPERSEDED:
        return "superseded"
    typed = await find_onboarding_interview_session(user_id)
    if typed is not None and interview_is_over(typed):
        return "superseded"
    seen: set[str] = set()
    current = await get_onboarding_voice_call_id(user_id)
    while current is not None and current != call_id and current not in seen:
        seen.add(current)
        record = await locks.read_call(redis, current)
        if record is None:
            # Written on the row at admit, before its record: starting while
            # it holds the person's lock.
            if await locks.live_call_of(redis, user_id) == current:
                return "waiting"
            return "clear"
        if record.uid != user_id:
            return "clear"
        words = call_words(record)
        if words == "words":
            return "superseded"
        if words == "unsettled":
            return "waiting"
        current = record.prev
    return "clear"


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

    The person's finish after their latest call ended is theirs to make
    over the call's words, as a message is (:func:`supersede_the_latest_call`).

    Raises:
        HTTPException: whatever :func:`refuse_while_a_call_holds_the_interview`
            and :func:`supersede_the_latest_call` raise while a voice call
            holds the interview: the step then shows the call, not the typed
            interview's ending.
    """
    await refuse_while_a_call_holds_the_interview(user_id)
    await supersede_the_latest_call(user_id)
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


def _own_corrections(corrections: str | None) -> str:
    """What the person changed on an ending screen, as the hand-off keeps it:
    without the platform's own tags, trimmed, within its limit."""
    return strip_server_injected_tags(corrections or "").strip()[:MAX_CORRECTION_CHARS]


def with_corrections(transcript: str, corrections: str | None) -> str:
    """The words to hand off: the interview's, then what the person changed
    on an ending screen, as their own words too.

    The corrections are their latest word, so they are kept within the typed
    take's limit and the interview's words are cut to make room before them.
    """
    extra = _own_corrections(corrections)
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
        return superseded_take()
    return FinalizeResponse(
        status=dump.status,
        input_mode=dump.inputMode,
        error_code=dump.errorCode,
    )
