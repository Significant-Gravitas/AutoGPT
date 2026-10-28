"""Authenticated, user-scoped onboarding brain-dump endpoints."""

import logging
from typing import Annotated

import autogpt_libs.auth as autogpt_auth_lib
import fastapi
from autogpt_libs.auth.dependencies import get_user_id
from fastapi import APIRouter, Form, Security, UploadFile
from fastapi.responses import RedirectResponse
from prisma.enums import BrainDumpInputMode, BrainDumpStatus

from backend.api.features.onboarding_dump import db, service, storage, suggested_tasks
from backend.api.features.onboarding_dump.models import (
    ALLOWED_AUDIO_MIME_TYPES,
    MAX_PART_BYTES,
    MAX_RECORDING_BYTES,
    RECORDING_ID_PATTERN,
    DumpStatusResponse,
    FinalizeRequest,
    FinalizeResponse,
    GreetingDoneResponse,
    IntroCardResponse,
    RecommendedExpertsResponse,
    RecommendedProvidersResponse,
    SuggestedTasksResponse,
    SuggestedTasksUpdate,
    UploadPartResponse,
)
from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError

logger = logging.getLogger(__name__)


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
    recording_id: Annotated[str, Form(pattern=RECORDING_ID_PATTERN)],
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


@router.get("/suggested-tasks", operation_id="get_brain_dump_suggested_tasks")
async def get_brain_dump_suggested_tasks(
    user_id: Annotated[str, Security(get_user_id)],
) -> SuggestedTasksResponse:
    """The greeting's suggested first tasks, each with its own state.

    Unlike ``/intro``, this keeps answering after the first message: the
    list stays on Home, a sent row marked in progress, until every row is
    dismissed.
    """
    return await suggested_tasks.get_suggested_tasks(user_id)


@router.post("/suggested-tasks", operation_id="update_brain_dump_suggested_tasks")
async def update_brain_dump_suggested_tasks(
    request: SuggestedTasksUpdate,
    user_id: Annotated[str, Security(get_user_id)],
) -> SuggestedTasksResponse:
    """Mark suggested tasks sent, dismissed, or open again (Undo).

    Takes several ids so "Hide all" and its Undo are one write each.
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
        str | None, fastapi.Query(pattern=RECORDING_ID_PATTERN)
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
