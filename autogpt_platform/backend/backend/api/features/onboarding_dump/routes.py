"""Authenticated, user-scoped onboarding brain-dump endpoints."""

import logging
from typing import Annotated

import autogpt_libs.auth as autogpt_auth_lib
import fastapi
from autogpt_libs.auth.dependencies import get_user_id
from fastapi import APIRouter, Form, Path, Security, UploadFile
from fastapi.responses import RedirectResponse
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from pydantic import AfterValidator

from backend.api.features.onboarding_dump import (
    db,
    interview,
    plan_hire,
    plan_service,
    service,
    storage,
    suggested_tasks,
    voice,
)
from backend.api.features.onboarding_dump.models import (
    ALLOWED_AUDIO_MIME_TYPES,
    MAX_PART_BYTES,
    MAX_RECORDING_BYTES,
    RECORDING_ID_PATTERN,
    VOICE_CALL_ID_PATTERN,
    DumpStatusResponse,
    FinalizeRequest,
    FinalizeResponse,
    GreetingDoneResponse,
    InterviewStartResponse,
    InterviewStateResponse,
    IntroCardResponse,
    RecommendedExpertsResponse,
    RecommendedProvidersResponse,
    SuggestedTasksResponse,
    SuggestedTasksUpdate,
    UploadPartResponse,
    VoiceCallFinishRequest,
    VoiceCallStartResponse,
    VoiceCallStatusResponse,
    clients_recording_id,
)
from backend.api.features.onboarding_dump.plan_models import (
    PlanHireResponse,
    PlanResponse,
)
from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.copilot.onboarding_interview import find_onboarding_interview_session
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)


def _clients_recording_id_or_none(recording_id: str | None) -> str | None:
    return None if recording_id is None else clients_recording_id(recording_id)


router = APIRouter(
    prefix="/onboarding/brain-dump",
    tags=["onboarding", "brain-dump"],
    dependencies=[
        Security(autogpt_auth_lib.requires_user),
    ],
)


@router.post("/parts", operation_id="upload_brain_dump_part")
async def upload_brain_dump_part(
    user_id: Annotated[str, Security(get_user_id)],
    file: UploadFile,
    recording_id: Annotated[
        str,
        Form(pattern=RECORDING_ID_PATTERN),
        AfterValidator(clients_recording_id),
    ],
    part_index: Annotated[int, Form(ge=0)],
    resent: Annotated[bool, Form()] = False,
) -> UploadPartResponse:
    """Accept one MediaRecorder timeslice blob.

    Parts arrive while the user is still talking, so this stays cheap:
    size checks, buffer, done. Validation that needs the whole stream
    (virus scan, transcription) happens on finalize.

    A part of a take the client discarded is refused, unless ``resent`` says
    the client is deliberately sending that take again from its local copy.
    """
    if file.content_type and file.content_type.split(";")[0] not in (
        ALLOWED_AUDIO_MIME_TYPES
    ):
        raise fastapi.HTTPException(
            status_code=415, detail=f"Unsupported audio type: {file.content_type}"
        )
    if not resent and await storage.was_discarded(user_id, recording_id):
        # A part already on the wire when the user skipped; letting part 0
        # through would reopen the take over the skip.
        raise fastapi.HTTPException(
            status_code=409, detail="This recording was discarded"
        )

    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(64 * 1024):
        total += len(chunk)
        if total > MAX_PART_BYTES:
            raise fastapi.HTTPException(
                status_code=413,
                detail=f"Recording part exceeds {MAX_PART_BYTES // (1024 * 1024)} MB",
            )
        chunks.append(chunk)

    if await storage.buffered_size(user_id, recording_id) + total > MAX_RECORDING_BYTES:
        raise fastapi.HTTPException(
            status_code=413,
            detail=(
                f"Recording exceeds the {MAX_RECORDING_BYTES // (1024 * 1024)} MB limit"
            ),
        )

    if part_index == 0:
        await db.start_dump(user_id, recording_id, BrainDumpInputMode.voice)
        # A discard that landed since the check above has already looked for
        # this take on the row and missed it, so give the row back here.
        if not resent and await storage.was_discarded(user_id, recording_id):
            await _release_discarded_take(user_id, recording_id)
            raise fastapi.HTTPException(
                status_code=409, detail="This recording was discarded"
            )

    cumulative = await storage.append_part(
        user_id, recording_id, part_index, b"".join(chunks)
    )
    # The check above races: two parts in flight can both read a buffer
    # that is under the cap and both be admitted. This one is
    # authoritative because the size comes back from inside the same
    # transaction that wrote the part. A recording past the ceiling is
    # not going to assemble into anything usable, so the buffer goes.
    if cumulative > MAX_RECORDING_BYTES:
        await storage.discard_parts(user_id, recording_id)
        raise fastapi.HTTPException(
            status_code=413,
            detail=(
                f"Recording exceeds the {MAX_RECORDING_BYTES // (1024 * 1024)} MB limit"
            ),
        )

    return UploadPartResponse(
        recording_id=recording_id,
        part_index=part_index,
        received_bytes=total,
        total_bytes=cumulative,
    )


@router.post("/finalize", operation_id="finalize_brain_dump")
async def finalize_brain_dump(
    request: FinalizeRequest,
    user_id: Annotated[str, Security(get_user_id)],
    background_tasks: fastapi.BackgroundTasks,
) -> FinalizeResponse:
    """Close out a take: assemble, scan, store, transcribe.

    Extraction and greeting generation continue in the background after
    this responds — the preparing step polls ``/status`` until
    ``greeting_ready``. Idempotent per ``recording_id`` — a client that
    retries after a timeout gets the stored result rather than a second
    transcription.
    """
    try:
        if request.input_mode == BrainDumpInputMode.skipped:
            return await service.finalize_skipped_dump(user_id, request.recording_id)

        if request.input_mode == BrainDumpInputMode.typed:
            text = (request.text or "").strip()
            if not text:
                raise fastapi.HTTPException(
                    status_code=422, detail="Typed brain dump cannot be empty"
                )
            return await service.finalize_typed_dump(
                user_id, request.recording_id, text, background_tasks
            )

        return await service.finalize_voice_dump(
            user_id,
            request.recording_id,
            request.duration_secs,
            request.mime_type,
            background_tasks,
        )
    except VirusDetectedError as e:
        await db.mark_failed(user_id, request.recording_id, "virus_detected")
        raise fastapi.HTTPException(status_code=400, detail=str(e)) from e
    except VirusScanError as e:
        await db.mark_failed(user_id, request.recording_id, "virus_scan_failed")
        raise fastapi.HTTPException(status_code=500, detail=str(e)) from e


@router.post("/interview", operation_id="start_onboarding_interview")
async def start_onboarding_interview(
    user_id: Annotated[str, Security(get_user_id)],
) -> InterviewStartResponse:
    """Open the person's onboarding interview, with Otto speaking first.

    The first call starts Otto's greeting; every later one adopts the same
    interview, so Back, a reload and a second tab land in one conversation.
    ``turn_id`` is set only when this call started the greeting. A greeting
    already being written is answered without one, like an interview under
    way: the client reads the session, which resumes the running turn. An
    interview that is over is answered the same way, with its ``stage``, so
    the client shows how it ended and the finish button.

    A person has one interview, typed or spoken: 409 ``call_in_progress``
    while a voice call of theirs is live, and 409
    ``onboarding_interview_closed`` once a call took the interview (its words
    accepted from a call that ran to its close, or that the person
    confirmed), each naming the call (``call_id``). A call that was cut off
    leaves the typed interview open. 429 at its turn
    budget, 503 while the budget, the chat stream or the person's calls
    cannot be read. A 429 carries its wait in the body
    (``retry_after_seconds``) as well as in ``Retry-After``: the browser's API
    client keeps a refusal's body, not its headers.
    """
    try:
        started = await interview.start_onboarding_interview(user_id)
    except fastapi.HTTPException as e:
        if e.status_code == 429:
            raise _with_wait_in_body(e) from e
        if e.status_code != 409 or e.detail != interview.TURN_IN_PROGRESS:
            raise
        # Only the greeting can be running before the person has said
        # anything, and the interview cannot be over yet.
        running = await find_onboarding_interview_session(user_id)
        if running is None:
            raise
        return InterviewStartResponse(session_id=running.session_id)
    return InterviewStartResponse(
        session_id=started.session.session_id,
        turn_id=started.turn_id,
        stage=started.session.metadata.onboarding_interview_stage,
    )


@router.get("/interview", operation_id="get_onboarding_interview")
async def get_onboarding_interview(
    user_id: Annotated[str, Security(get_user_id)],
) -> InterviewStateResponse:
    """The person's onboarding interview as it stands, without starting it,
    and whether the step may offer them a call.

    For the step's talk-or-type choice, which must know whether the
    interview is over before it offers Skip, and shows "Talk to Otto" only
    where voice can be offered. Nothing is created, greeted, charged or
    refused. An interview that is over but not yet handed off is handed off
    in the background, as the start does. A person who has not opened their
    interview reads with no session.
    """
    read = await interview.read_onboarding_interview(user_id)
    voice_available = await voice.voice_available(user_id)
    livekit_url = voice.livekit_url() if voice_available else None
    if read is None:
        return InterviewStateResponse(
            session_id=None,
            is_over=False,
            voice_available=voice_available,
            livekit_url=livekit_url,
        )
    return InterviewStateResponse(
        session_id=read.session.session_id,
        stage=read.session.metadata.onboarding_interview_stage,
        is_over=read.is_over,
        voice_available=voice_available,
        livekit_url=livekit_url,
    )


def _with_wait_in_body(refusal: fastapi.HTTPException) -> fastapi.HTTPException:
    """*refusal* with its message and its ``Retry-After`` wait in the body."""
    retry_after = (refusal.headers or {}).get("Retry-After", "")
    return fastapi.HTTPException(
        status_code=refusal.status_code,
        detail={
            "message": refusal.detail,
            "retry_after_seconds": int(retry_after) if retry_after.isdigit() else None,
        },
        headers=refusal.headers,
    )


@router.post("/interview/voice", operation_id="start_onboarding_voice_call")
async def start_onboarding_voice_call(
    user_id: Annotated[str, Security(get_user_id)],
) -> VoiceCallStartResponse:
    """Start a voice call for the person's onboarding interview.

    Answers with the call's room, which carries its audio, and the browser's
    token for it, and with the relay's call socket and a start token good for
    this one call within a minute. The browser joins the room, then opens the
    socket and sends the token as its first message; a browser that cannot
    join the room ends the call (``/end``). The call counts against the
    person's allowance once its model session opens on the relay.

    404 while voice is off for the person; 409 once their interview is over,
    or while a call of theirs is live (``call_id`` names it); 429 past their
    call allowance; 503 past the platform's voice budget or live-call
    ceiling, or while voice cannot be served. A 429 or 503 carries its wait
    in the body (``retry_after_seconds``) as well as in ``Retry-After``.
    """
    try:
        return await voice.start_voice_call(user_id)
    except fastapi.HTTPException as e:
        if e.status_code in (429, 503):
            raise _with_wait_in_body(e) from e
        raise


VoiceCallId = Annotated[str, Path(pattern=VOICE_CALL_ID_PATTERN)]


@router.get("/interview/voice", operation_id="get_latest_onboarding_voice_call")
async def get_latest_onboarding_voice_call(
    user_id: Annotated[str, Security(get_user_id)],
) -> VoiceCallStatusResponse:
    """The person's latest voice call as it stands, for a step that reloads:
    live, wrapping up, waiting for their confirm, handed off or failed. 404
    when they have no call, or its record has expired."""
    try:
        return await voice.latest_call(user_id)
    except fastapi.HTTPException as e:
        if e.status_code == 503:
            raise _with_wait_in_body(e) from e
        raise


@router.get(
    "/interview/voice/{call_id}", operation_id="get_onboarding_voice_call_status"
)
async def get_onboarding_voice_call_status(
    user_id: Annotated[str, Security(get_user_id)],
    call_id: VoiceCallId,
) -> VoiceCallStatusResponse:
    """A voice call of the person's as it stands, polled once it ends until
    its hand-off has answered. Its ``interview`` says whether the call's words
    took the person's interview or what they typed after the call superseded
    them, and is read before ``end_reason`` and ``status``. 404 for a call
    that is not theirs, or has expired."""
    try:
        return await voice.call_status(user_id, call_id)
    except fastapi.HTTPException as e:
        if e.status_code == 503:
            raise _with_wait_in_body(e) from e
        raise


@router.post(
    "/interview/voice/{call_id}/finish", operation_id="finish_onboarding_voice_call"
)
async def finish_onboarding_voice_call(
    user_id: Annotated[str, Security(get_user_id)],
    call_id: VoiceCallId,
    background_tasks: fastapi.BackgroundTasks,
    request: VoiceCallFinishRequest | None = None,
) -> FinalizeResponse:
    """The person's confirm of their voice call, and "Meet your team".

    Idempotent: a call that took the interview or that another take beat, or
    whose words the pipeline ruled on, is answered as it stands, and a call
    the person's typing superseded is answered ``superseded``. A call waiting
    for the confirm, one whose hand-off the pipeline never ruled on, or one
    cut off with its words accepted hands its words off now, with any
    ``corrections`` after them, and its accepted take becomes the interview's.
    The client reads the answer as it reads the typed interview's finish. 404
    for a call that is not theirs; 409
    (``call_not_ready``) while the call's words are not ready. A 429 or 503
    carries its wait in the body (``retry_after_seconds``).
    """
    try:
        return await voice.finish_call(
            user_id,
            call_id,
            background_tasks,
            corrections=request.corrections if request is not None else None,
        )
    except fastapi.HTTPException as e:
        if e.status_code in (429, 503):
            raise _with_wait_in_body(e) from e
        raise


@router.post(
    "/interview/voice/{call_id}/end",
    operation_id="end_onboarding_voice_call",
    status_code=202,
)
async def end_onboarding_voice_call(
    user_id: Annotated[str, Security(get_user_id)],
    call_id: VoiceCallId,
) -> None:
    """End a live voice call of the person's that they can no longer reach,
    such as one on another device, or one whose start they gave up on. The
    relay running it ends it within a second or so, and the person can call
    again. 404 for a call that is not theirs."""
    try:
        await voice.end_call(user_id, call_id)
    except fastapi.HTTPException as e:
        if e.status_code == 503:
            raise _with_wait_in_body(e) from e
        raise


@router.post("/interview/finish", operation_id="finish_onboarding_interview")
async def finish_onboarding_interview(
    user_id: Annotated[str, Security(get_user_id)],
    background_tasks: fastapi.BackgroundTasks,
) -> FinalizeResponse:
    """The person ends the interview on screen: close it if it is still
    open, and hand what they said to the onboarding pipeline.

    Idempotent: an interview already handed off is not handed off again, and
    the response describes its take as it stands. A take that failed without
    a ruling on the words (``quality_check_unavailable``, or no code) leaves
    the interview to be handed off again on the next call. The pipeline's
    jobs run after the response, as a typed brain dump's do. 404 when the
    person has no interview. As the start route does, 409 ``call_in_progress``
    while a voice call of theirs is live and 409
    ``onboarding_interview_closed`` once a call took the interview, each
    naming the call (``call_id``). A 429 carries its wait in the body
    (``retry_after_seconds``), as the start route's does.
    """
    opened = await find_onboarding_interview_session(user_id)
    if opened is None:
        raise fastapi.HTTPException(status_code=404, detail="No onboarding interview")
    try:
        return await interview.finalize_onboarding_interview(
            user_id, opened.session_id, background_tasks
        )
    except NotFoundError as e:
        raise fastapi.HTTPException(
            status_code=404, detail="No onboarding interview"
        ) from e
    except fastapi.HTTPException as e:
        if e.status_code == 429:
            raise _with_wait_in_body(e) from e
        raise


@router.get("/status", operation_id="get_brain_dump_status")
async def get_brain_dump_status(
    user_id: Annotated[str, Security(get_user_id)],
) -> DumpStatusResponse:
    """Polled by the loading screen while the pipeline runs."""
    dump = await db.get_dump(user_id)
    if dump is None:
        return DumpStatusResponse()
    return DumpStatusResponse(
        status=dump.status,
        input_mode=dump.inputMode,
        error_code=dump.errorCode,
        has_audio=bool(dump.audioPath),
        # Skipped dumps get the static Path B greeting, so there is
        # nothing to generate or wait for.
        greeting_ready=(
            dump.inputMode == BrainDumpInputMode.skipped
            or bool((dump.greeting or "").strip())
        ),
    )


@router.get("/intro", operation_id="get_brain_dump_intro")
async def get_brain_dump_intro(
    user_id: Annotated[str, Security(get_user_id)],
) -> IntroCardResponse:
    """Content for the copilot home's onboarding greeting.

    Pre-generated during finalize, so this is a plain row read — the
    copilot home never waits on a model to render its first screen.
    ``greeting_done=true`` means the client should show nothing and cache
    that locally.
    """
    return await service.get_intro_card(user_id)


@router.get(
    "/recommended-providers", operation_id="get_brain_dump_recommended_providers"
)
async def get_brain_dump_recommended_providers(
    user_id: Annotated[str, Security(get_user_id)],
) -> RecommendedProvidersResponse:
    """Provider picks for the welcome dialog's "Connect your tools" panel.

    Written by a background job that runs beside the greeting one; a plain
    row read here. ``ready=false`` means keep polling.
    """
    return await service.get_recommended_providers(user_id)


@router.get("/recommended-experts", operation_id="get_brain_dump_recommended_experts")
async def get_brain_dump_recommended_experts(
    user_id: Annotated[str, Security(get_user_id)],
) -> RecommendedExpertsResponse:
    """The team Otto proposes: who to hire first, and the raise door.

    Written by a background job beside the greeting one; a plain read
    here, with the deterministic fallback filled in for takes the job
    never ran on. ``ready=false`` means keep polling. ``team`` is null
    when the expert-team feature is off for this user.
    """
    return await service.get_recommended_experts(user_id)


@router.get("/plan", operation_id="get_brain_dump_plan")
async def get_brain_dump_plan(
    user_id: Annotated[str, Security(get_user_id)],
) -> PlanResponse:
    """The onboarding plan step: what Otto heard, the team, the ranked tasks.

    ``ready=false`` means the background jobs are still running; keep
    polling. Every other state resolves, a skipped or failed dump to the
    role plan, which is stored on this first read.
    """
    return await plan_service.get_plan(user_id)


@router.post("/plan/settle", operation_id="settle_brain_dump_plan")
async def settle_brain_dump_plan(
    user_id: Annotated[str, Security(get_user_id)],
) -> PlanResponse:
    """The plan step stops waiting: the plan it shows becomes the plan.

    While the background jobs are still running, the plan is stored as it
    stands: the greeting's own tasks once it has written them, with the team
    the team job chose (or the role team while it is still choosing), and
    the role plan before that. The trial then hires, and Home lists, what
    the step showed; the jobs leave both alone. Once a plan is ready this
    changes nothing and returns it.
    """
    return await plan_service.settle_plan(user_id)


@router.post(
    "/plan/hire",
    operation_id="hire_brain_dump_plan",
    dependencies=[fastapi.Depends(enforce_payment_paywall)],
)
async def hire_brain_dump_plan(
    user_id: Annotated[str, Security(get_user_id)],
) -> PlanHireResponse:
    """Hire the plan's team, after the trial is confirmed.

    Behind the paywall, so a user who left Stripe without paying hires
    nobody, whatever the client does; with no paywall it is open.

    Idempotent: anyone already on the team is reported, not hired again.
    A hire that fails is reported in ``failed`` rather than failing the
    request, and the active expert limit stops the hiring where it is hit.
    """
    return await plan_hire.hire_plan(user_id)


@router.get("/suggested-tasks", operation_id="get_brain_dump_suggested_tasks")
async def get_brain_dump_suggested_tasks(
    user_id: Annotated[str, Security(get_user_id)],
) -> SuggestedTasksResponse:
    """The greeting's suggested first tasks, each with its own state.

    Unlike ``/intro``, this keeps answering after the first message: the
    list stays on Home, a sent row marked in progress, until every row is
    dismissed. Each task names its owner's hired expert, if any.
    """
    return await suggested_tasks.get_suggested_tasks(user_id)


@router.post("/suggested-tasks", operation_id="update_brain_dump_suggested_tasks")
async def update_brain_dump_suggested_tasks(
    request: SuggestedTasksUpdate,
    user_id: Annotated[str, Security(get_user_id)],
) -> SuggestedTasksResponse:
    """Mark suggested tasks sent, dismissed, or open again (Undo).

    Takes several ids so a change and its Undo are one write each.
    """
    return await suggested_tasks.update_suggested_tasks(user_id, request)


@router.post("/intro/complete", operation_id="complete_brain_dump_greeting")
async def complete_brain_dump_greeting(
    user_id: Annotated[str, Security(get_user_id)],
) -> GreetingDoneResponse:
    """Mark the greeting as done — called when the first message is sent.

    Permanent and content-preserving: the flag flips, the stored greeting
    stays in the row.
    """
    await service.mark_greeting_done(user_id)
    return GreetingDoneResponse(greeting_done=True)


@router.get("/recording", operation_id="download_brain_dump_recording")
async def download_brain_dump_recording(
    user_id: Annotated[str, Security(get_user_id)],
) -> RedirectResponse:
    """Hand back the user's own recording, for as long as it is kept.

    Meant for a take that failed: "your recording is safe" has to be
    true, and this is what makes it checkable. A transcribed take has no
    recording left, since only the transcript is kept. A failed take's
    recording is deleted within 48 hours (see ``retention.py``), and one
    that is due is refused even before the sweep reaches it; a signed URL
    for one that is gone would only fail at the bucket, so say so here
    instead.
    """
    dump = await db.get_dump(user_id)
    if dump is None or not dump.audioPath:
        raise fastapi.HTTPException(status_code=404, detail="No recording found")
    try:
        available = await storage.audio_available(dump.audioPath)
    except Exception as e:
        # Type only: a GCS error carries the object URL, which holds the
        # user ID, and so would the traceback of a 500.
        logger.warning(
            f"Brain dump recording check failed at the bucket: {type(e).__name__}"
        )
        raise fastapi.HTTPException(
            status_code=503, detail="Recording temporarily unavailable, try again"
        ) from None
    if not available:
        # So ``has_audio`` stops offering it, even if the blob went some way
        # other than the sweep, which clears the row itself.
        await db.clear_audio_paths([dump.audioPath])
        raise fastapi.HTTPException(
            status_code=404, detail="Recording no longer available"
        )

    url = await storage.audio_download_url(user_id, dump.audioPath)
    return RedirectResponse(url=url, status_code=307)


# Path is "" rather than "/": with the router prefix that would register
# ``/api/onboarding/brain-dump/`` and every call to the unslashed URL would
# take a 307 through FastAPI's redirect_slashes. Redirects drop the
# Authorization header, so the retried request arrives unauthenticated and
# 401s. No other route in this codebase uses "/" — this is why.
@router.delete("", operation_id="discard_brain_dump")
async def discard_brain_dump(
    user_id: Annotated[str, Security(get_user_id)],
    recording_id: Annotated[
        str | None,
        fastapi.Query(pattern=RECORDING_ID_PATTERN),
        AfterValidator(_clients_recording_id_or_none),
    ] = None,
) -> DumpStatusResponse:
    """Drop a half-uploaded take's server buffer when the user re-records.

    Stored audio is left alone. A named take still uploading is also marked
    failed, so it no longer holds the row against the user's next take, and
    a part of it that arrives later is refused.

    ``recording_id`` says *which* take to drop. Without it the row's
    current take is assumed, which is wrong as soon as a second tab has
    moved the row on: the caller would clear a buffer that is still being
    filled. Callers know their own id, so they send it.
    """
    # Mark before releasing: a late part 0 claims the row and then reads the
    # mark, so one of the two always sees the other. Both before the buffer goes.
    if recording_id:
        try:
            await storage.mark_discarded(user_id, recording_id)
        finally:
            await _release_discarded_take(user_id, recording_id)
    dump = await db.get_dump(user_id)
    target = recording_id or (dump.recordingId if dump else None)
    if target and not (
        dump is not None
        and dump.recordingId == target
        and dump.status == BrainDumpStatus.completed
    ):
        await storage.discard_parts(user_id, target)
    return DumpStatusResponse(
        status=dump.status if dump else None,
        input_mode=dump.inputMode if dump else None,
        has_audio=bool(dump and dump.audioPath),
    )


async def _release_discarded_take(user_id: str, recording_id: str) -> None:
    """Fail a discarded take that is still waiting for parts, so it stops holding the row."""
    await db.claim_transition(
        user_id,
        recording_id,
        expected=BrainDumpStatus.recording_uploaded,
        new=BrainDumpStatus.failed,
        errorCode="discarded",
    )
