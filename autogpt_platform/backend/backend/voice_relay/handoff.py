"""A call's hand-off: its transcript of record to the person's interview, and
the answer on the call's record.

Shared by the relay's post-call job (``postcall``), which hands a call off
once its transcript is made, and the REST API's finish route
(``onboarding_dump.voice``), which hands it off on the person's confirm or
retry. The call's status follows the answer: ``handed_off`` once the pipeline
took the words or another take owns the interview, ``failed`` with the answer
while the interview stays open, or ``transcribed`` while a confirm arm waits
for the person.

A call whose words the pipeline accepted takes the person's interview when it
ran to its close (:data:`RAN_TO_ITS_CLOSE`), or when the person confirmed it.
A call that was cut off is handed off all the same, so the person has a Home
if they never come back, but leaves the interview open: a later call, or the
typed interview, replaces its take as any re-take does. Which call took the
interview is written here, where the hand-off answers: on the person's
onboarding row, which outlives the call, and on the call's record.
"""

from fastapi import BackgroundTasks, HTTPException
from prisma.enums import BrainDumpInputMode, BrainDumpStatus

from backend.api.features.onboarding_dump.interview import (
    InterviewClosure,
    close_voice_interview,
    voice_hand_off_closes_interview,
)
from backend.api.features.onboarding_dump.models import FinalizeResponse
from backend.copilot.onboarding_interview import InterviewEndReason
from backend.data.onboarding import record_onboarding_voice_interview_call
from backend.data.redis_client import AsyncRedisClient
from backend.voice_relay import locks
from backend.voice_relay.locks import CallRecord, CallStatus

# How a call ends when it runs to its close: Otto's goodbye, the server's caps,
# the scope rule, or the person's Hang up. Every other end cut the call off:
# the connection, silence, the vendor, a pod, or "End that call" from another
# device.
RAN_TO_ITS_CLOSE = frozenset(
    {"finished", "wall_clock", "turn_cap", "spend_cap", "scope", "user_hangup"}
)

# The pipeline's refusals to take a hand-off for now, as a call records them:
# over the person's personalization budget, or the budget cannot be read.
RATE_LIMITED = "rate_limited"
PERSONALIZATION_UNAVAILABLE = "personalization_unavailable"

# The interview's reasons for the relay's: why the call ended, as the
# interview words it. The server ending a call it could not keep counts as
# the cap.
_INTERVIEW_REASONS: dict[str, InterviewEndReason] = {
    "finished": "confirmed",
    "wall_clock": "cap",
    "turn_cap": "cap",
    "spend_cap": "cap",
    "idle": "idle",
    "scope": "scope",
    "user_hangup": "user",
    "superseded": "user",
    "client_gone": "user",
}


def interview_reason(end_reason: str | None) -> InterviewEndReason:
    return _INTERVIEW_REASONS.get(end_reason or "", "cap")


def ran_to_its_close(end_reason: str | None) -> bool:
    """Whether a call that ended with *end_reason* ran to its close, rather
    than being cut off. An end this code does not know counts as cut off."""
    return end_reason in RAN_TO_ITS_CLOSE


def refused_hand_off(refusal: HTTPException) -> FinalizeResponse | None:
    """The pipeline's refusal to take a hand-off now, as the answer a call
    records. None for any other error."""
    code = {429: RATE_LIMITED, 503: PERSONALIZATION_UNAVAILABLE}.get(
        refusal.status_code
    )
    if code is None:
        return None
    return failed_hand_off(code)


def failed_hand_off(code: str) -> FinalizeResponse:
    return FinalizeResponse(
        status=BrainDumpStatus.failed,
        input_mode=BrainDumpInputMode.typed,
        error_code=code,
    )


def call_status_after(closure: InterviewClosure) -> CallStatus:
    """The call's status once its hand-off has answered."""
    if closure.hand_off is None:
        return locks.TRANSCRIBED
    if voice_hand_off_closes_interview(closure.hand_off):
        return locks.HANDED_OFF
    return locks.FAILED


async def record_hand_off(
    redis: AsyncRedisClient,
    call_id: str,
    closure: InterviewClosure,
    *,
    took_interview: bool = False,
) -> CallStatus:
    """Record a hand-off's answer on the call, and whether the call took the
    interview with it. Returns the call's status."""
    status = call_status_after(closure)
    hand_off = closure.hand_off
    await locks.forget_call_fields(
        redis,
        call_id,
        {
            "status": status,
            "ending": closure.ending,
            "hand_off": hand_off.model_dump(mode="json") if hand_off else None,
            "took_interview": True if took_interview else None,
        },
        *(() if hand_off else ("hand_off",)),
    )
    return status


def takes_the_interview(
    closure: InterviewClosure, record: CallRecord, *, confirmed: bool
) -> bool:
    """Whether a hand-off's answer makes the call the interview's take: the
    pipeline accepted its words, and the call ran to its close or the person
    confirmed it."""
    hand_off = closure.hand_off
    accepted = hand_off is not None and hand_off.status != BrainDumpStatus.failed
    return accepted and (confirmed or ran_to_its_close(record.end_reason))


async def hand_off_call(
    redis: AsyncRedisClient,
    record: CallRecord,
    background_tasks: BackgroundTasks,
    *,
    confirmed: bool = False,
    corrections: str | None = None,
) -> InterviewClosure:
    """Hand a call's stored transcript of record to the person's interview,
    and record the answer on the call. The pipeline refusing it for now (429
    or 503) is recorded as a failed answer, then raised.

    *confirmed* is the person's own finish of the call, which makes an
    accepted take the interview's whatever ended the call. The call that took
    the interview is written on the person's onboarding row before the call's
    record, so the record never says so while the row does not.
    """
    try:
        closure = await close_voice_interview(
            record.uid,
            record.call_id,
            record.transcript_of_record or "",
            background_tasks,
            reason=interview_reason(record.end_reason),
            confirmed=confirmed,
            corrections=corrections,
            first_task=record.first_task,
        )
    except HTTPException as refusal:
        answer = refused_hand_off(refusal)
        if answer is not None:
            await record_hand_off(
                redis,
                record.call_id,
                InterviewClosure(ending=None, hand_off=answer),
            )
        raise
    took_interview = takes_the_interview(closure, record, confirmed=confirmed)
    if took_interview:
        await record_onboarding_voice_interview_call(record.uid, record.call_id)
    await record_hand_off(
        redis, record.call_id, closure, took_interview=took_interview
    )
    return closure
