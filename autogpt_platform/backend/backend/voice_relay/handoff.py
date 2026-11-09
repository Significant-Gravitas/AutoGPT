"""A call's hand-off: its transcript of record to the person's interview, and
the answer on the call's record.

Shared by the relay's post-call job (``postcall``), which hands a call off
once its transcript is made, and the REST API's finish route
(``onboarding_dump.voice``), which hands it off on the person's confirm or
retry. The call's status follows the answer: ``handed_off`` once it closed
the interview, ``failed`` with the answer while the interview stays open, or
``transcribed`` while a confirm arm waits for the person.
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
from backend.data.redis_client import AsyncRedisClient
from backend.voice_relay import locks
from backend.voice_relay.locks import CallRecord, CallStatus

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
    redis: AsyncRedisClient, call_id: str, closure: InterviewClosure
) -> CallStatus:
    """Record a hand-off's answer on the call. Returns the call's status."""
    status = call_status_after(closure)
    hand_off = closure.hand_off
    await locks.forget_call_fields(
        redis,
        call_id,
        {
            "status": status,
            "ending": closure.ending,
            "hand_off": hand_off.model_dump(mode="json") if hand_off else None,
        },
        *(() if hand_off else ("hand_off",)),
    )
    return status


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
    or 503) is recorded as a failed answer, then raised."""
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
    await record_hand_off(redis, record.call_id, closure)
    return closure
