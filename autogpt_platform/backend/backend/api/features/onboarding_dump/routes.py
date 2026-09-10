"""Onboarding brain-dump endpoints.

Everything here is user-scoped and gated on ``ONBOARDING_BRAIN_DUMP`` — a
stale frontend must not be able to start writing recordings after the flag
is turned back off.
"""

import logging
from typing import Annotated

import autogpt_libs.auth as autogpt_auth_lib
import fastapi
from autogpt_libs.auth.dependencies import get_user_id
from fastapi import APIRouter, Depends, Form, Security, UploadFile
from fastapi.responses import RedirectResponse
from prisma.enums import BrainDumpInputMode, BrainDumpStatus

from backend.api.features.onboarding_dump import db, service, storage
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
    RecommendedProvidersResponse,
    UploadPartResponse,
)
from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)


async def require_brain_dump_flag(
    user_id: Annotated[str, Security(get_user_id)],
) -> None:
    """Gate every endpoint on the brain-dump flag, fail-closed.

    Deliberately not ``create_feature_flag_dependency``: that helper
    short-circuits to 404 whenever LaunchDarkly has no SDK key, which is
    every local dev and test environment — and it does so *before*
    consulting the ``FORCE_FLAG_ONBOARDING_BRAIN_DUMP`` override, so the
    documented way to switch the feature on locally could never work.
    ``is_feature_enabled`` checks the env override first and still
    defaults to False when LaunchDarkly is unreachable.
    """
    if not await is_feature_enabled(Flag.ONBOARDING_BRAIN_DUMP, user_id):
        raise fastapi.HTTPException(status_code=404, detail="Feature not available")


router = APIRouter(
    prefix="/onboarding/brain-dump",
    tags=["onboarding", "brain-dump"],
    dependencies=[
        Security(autogpt_auth_lib.requires_user),
        Depends(require_brain_dump_flag),
    ],
)


@router.post("/parts", operation_id="upload_brain_dump_part")
async def upload_brain_dump_part(
    user_id: Annotated[str, Security(get_user_id)],
    file: UploadFile,
    recording_id: Annotated[str, Form(pattern=RECORDING_ID_PATTERN)],
    part_index: Annotated[int, Form(ge=0)],
) -> UploadPartResponse:
    """Accept one MediaRecorder timeslice blob.

    Parts arrive while the user is still talking, so this stays cheap:
    size checks, buffer, done. Validation that needs the whole stream
    (virus scan, transcription) happens on finalize.
    """
    if file.content_type and file.content_type.split(";")[0] not in (
        ALLOWED_AUDIO_MIME_TYPES
    ):
        raise fastapi.HTTPException(
            status_code=415, detail=f"Unsupported audio type: {file.content_type}"
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
                "Recording exceeds the "
                f"{MAX_RECORDING_BYTES // (1024 * 1024)} MB limit"
            ),
        )

    if part_index == 0:
        await db.start_dump(user_id, recording_id, BrainDumpInputMode.voice)

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
                "Recording exceeds the "
                f"{MAX_RECORDING_BYTES // (1024 * 1024)} MB limit"
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
    """Hand back the user's own recording.

    Kept working even when ``status`` is ``failed`` — "your recording is
    safe" has to be true, and this is what makes it checkable.
    """
    dump = await db.get_dump(user_id)
    if dump is None or not dump.audioPath:
        raise fastapi.HTTPException(status_code=404, detail="No recording found")

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

    The row and any stored audio are left alone — only the disposable
    part buffer is cleared.

    ``recording_id`` says *which* take to drop. Without it the row's
    current take is assumed, which is wrong as soon as a second tab has
    moved the row on: the caller would clear a buffer that is still being
    filled. Callers know their own id, so they send it.
    """
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
