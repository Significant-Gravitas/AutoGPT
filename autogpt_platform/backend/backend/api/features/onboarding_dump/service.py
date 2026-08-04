"""The brain-dump pipeline: parts → audio → transcript → understanding.

Every step advances ``OnboardingBrainDump.status`` so the loading screen
can narrate honestly, and no failure ever deletes the audio — a failed
dump stays downloadable and retryable.
"""

import asyncio
import logging
import time

from fastapi import BackgroundTasks
from prisma import Json
from prisma.enums import BrainDumpInputMode, BrainDumpStatus

from backend.api.features.onboarding_dump import (
    db,
    intro,
    prompts,
    recommend,
    storage,
    transcription,
)
from backend.api.features.onboarding_dump.models import (
    FinalizeResponse,
    IntroCardResponse,
    RecommendedProvider,
    RecommendedProvidersResponse,
    SuggestedPrompt,
)
from backend.data.onboarding import format_brain_dump_for_extraction
from backend.data.tally import extract_business_understanding
from backend.data.understanding import (
    BusinessUnderstandingInput,
    get_business_understanding,
    upsert_business_understanding,
)
from backend.util.virus_scanner import scan_content_safe

logger = logging.getLogger(__name__)

TRANSCRIPT_PREVIEW_CHARS = 280


async def finalize_voice_dump(
    user_id: str,
    recording_id: str,
    duration_secs: float | None,
    mime_type: str | None,
    background_tasks: BackgroundTasks,
) -> FinalizeResponse:
    existing = await db.get_dump(user_id)
    if (
        existing is not None
        and existing.recordingId == recording_id
        and existing.status
        in (
            # Also idempotent while the background half of the pipeline is
            # still running: a client retry must not re-transcribe (the
            # part buffer is already discarded) or double-run extraction.
            #
            # ``transcribing`` matters most of all. The buffer is dropped
            # the moment the audio is stored, so a retry that arrives
            # while a long recording is still being transcribed would
            # assemble nothing and mark a perfectly good take failed.
            BrainDumpStatus.transcribing,
            BrainDumpStatus.transcribed,
            BrainDumpStatus.extracting,
            BrainDumpStatus.completed,
        )
    ):
        return _pipeline_response(
            existing.status, existing.transcript, BrainDumpInputMode.voice
        )

    # The guard above is a read followed by a write, so two finalizes
    # that arrive together can both pass it and both go on to assemble,
    # store and transcribe the same take. This claim is decided by the
    # database, so exactly one of them proceeds.
    claimed = await db.claim_transition(
        user_id,
        recording_id,
        expected=BrainDumpStatus.recording_uploaded,
        new=BrainDumpStatus.transcribing,
        errorCode=None,
    )
    if not claimed:
        current = await db.get_dump(user_id)
        if current is None:
            # No row, so there is nothing to write results to: every
            # ``update_dump`` below would no-op while ``discard_parts``
            # dropped the buffer for real, leaving stored audio nobody
            # can reach and a client told its dump succeeded. Stop here
            # and keep the parts — the browser still holds them, and a
            # re-upload recreates the row on part 0.
            return FinalizeResponse(
                status=BrainDumpStatus.failed,
                input_mode=BrainDumpInputMode.voice,
                error_code="no_audio_received",
            )
        if current.recordingId != recording_id:
            # There is one row per user, so a newer take from a second
            # tab has taken it over. Returning its status would narrate
            # somebody else's recording back to this client, and writing
            # anything would knock the live take over — so report this
            # take's own outcome and leave the row alone.
            return FinalizeResponse(
                status=BrainDumpStatus.failed,
                input_mode=BrainDumpInputMode.voice,
                error_code="superseded",
            )
        # Our take, already moving under another request. Not ours to
        # process, but its status is the honest answer.
        return _pipeline_response(
            current.status, current.transcript, BrainDumpInputMode.voice
        )

    audio = await storage.assemble_parts(user_id, recording_id)
    if not audio:
        # Nothing buffered server-side. The browser still holds every part
        # in IndexedDB, so this is a re-upload prompt, not a lost dump.
        await db.mark_failed(user_id, recording_id, "no_audio_received")
        return FinalizeResponse(
            status=BrainDumpStatus.failed,
            input_mode=BrainDumpInputMode.voice,
            error_code="no_audio_received",
        )

    filename = _audio_filename(recording_id, mime_type)
    await scan_content_safe(audio, filename=filename)
    try:
        audio_path = await storage.store_audio(user_id, audio, filename)
    except Exception as e:
        logger.warning("Brain dump audio storage failed for user %s: %s", user_id, e)
        await db.mark_failed(user_id, recording_id, "storage_failed")
        return FinalizeResponse(
            status=BrainDumpStatus.failed,
            input_mode=BrainDumpInputMode.voice,
            error_code="storage_failed",
        )

    await db.update_dump(
        user_id,
        recording_id,
        status=BrainDumpStatus.transcribing,
        audioPath=audio_path,
        mimeType=mime_type,
        sizeBytes=len(audio),
        durationSecs=duration_secs,
        errorCode=None,
    )
    # The audio is durable now; the buffer has done its job.
    await storage.discard_parts(user_id, recording_id)

    started_at = time.monotonic()
    try:
        result = await transcription.transcribe(audio, filename, duration_secs)
    except Exception as e:
        logger.warning("Brain dump transcription failed for user %s: %s", user_id, e)
        error_code = (
            "transcription_unavailable"
            if isinstance(e, transcription.TranscriptionUnavailableError)
            else "transcription_failed"
        )
        await db.mark_failed(user_id, recording_id, error_code)
        return FinalizeResponse(
            status=BrainDumpStatus.failed,
            input_mode=BrainDumpInputMode.voice,
            error_code=error_code,
        )

    logger.info(
        "Brain dump transcribed for user %s in %.1fs (%s chars)",
        user_id,
        time.monotonic() - started_at,
        len(result.text),
    )
    await db.update_dump(
        user_id,
        recording_id,
        status=BrainDumpStatus.transcribed,
        transcript=result.text,
        transcriptLang=result.language,
    )
    # Finalize only does the audio work. Extraction and the Sonnet
    # greeting are LLM calls that can outlive the frontend proxy's 30s
    # request timeout, so they run after this response is sent — the
    # preparing step polls /status and holds the user until
    # ``greeting_ready`` flips.
    background_tasks.add_task(
        _run_background_jobs,
        user_id,
        recording_id,
        result.text,
        BrainDumpInputMode.voice,
    )
    return _pipeline_response(
        BrainDumpStatus.transcribed, result.text, BrainDumpInputMode.voice
    )


async def finalize_typed_dump(
    user_id: str,
    recording_id: str,
    text: str,
    background_tasks: BackgroundTasks,
) -> FinalizeResponse:
    # Same idempotency contract as the voice path: a repeated submit must
    # not reset the row or queue a second extraction and greeting on top
    # of the pair already running.
    existing = await db.get_dump(user_id)
    if (
        existing is not None
        and existing.recordingId == recording_id
        and existing.status
        in (
            BrainDumpStatus.transcribing,
            BrainDumpStatus.transcribed,
            BrainDumpStatus.extracting,
            BrainDumpStatus.completed,
        )
    ):
        return _pipeline_response(
            existing.status, existing.transcript, existing.inputMode
        )

    await db.start_dump(user_id, recording_id, BrainDumpInputMode.typed)
    # Same reasoning as the voice path: only the caller that wins this
    # transition queues the extraction and greeting pair.
    claimed = await db.claim_transition(
        user_id,
        recording_id,
        expected=BrainDumpStatus.recording_uploaded,
        new=BrainDumpStatus.transcribed,
        transcript=text.strip(),
    )
    if not claimed:
        # Losing the claim means another finalize owns the row — this
        # take, or a newer one submitted from a second tab. Either way
        # this request must not queue a second pipeline, so there is no
        # falling through to the tasks below.
        current = await db.get_dump(user_id)
        if current is None:
            return FinalizeResponse(
                status=BrainDumpStatus.failed, input_mode=BrainDumpInputMode.typed
            )
        if current.recordingId != recording_id:
            return FinalizeResponse(
                status=BrainDumpStatus.failed,
                input_mode=BrainDumpInputMode.typed,
                error_code="superseded",
            )
        return _pipeline_response(current.status, current.transcript, current.inputMode)

    background_tasks.add_task(
        _run_background_jobs,
        user_id,
        recording_id,
        text.strip(),
        BrainDumpInputMode.typed,
    )
    return _pipeline_response(
        BrainDumpStatus.transcribed, text.strip(), BrainDumpInputMode.typed
    )


async def finalize_skipped_dump(user_id: str, recording_id: str) -> FinalizeResponse:
    """Record the skip so the copilot intro can take Path B.

    Skipping is a first-class outcome, not an error — the row exists so
    the intro endpoint can tell "skipped" apart from "never got here".
    """
    await db.start_dump(user_id, recording_id, BrainDumpInputMode.skipped)
    await db.update_dump(user_id, recording_id, status=BrainDumpStatus.completed)
    return FinalizeResponse(
        status=BrainDumpStatus.completed, input_mode=BrainDumpInputMode.skipped
    )


async def _run_background_jobs(
    user_id: str,
    recording_id: str,
    transcript: str,
    input_mode: BrainDumpInputMode,
) -> None:
    """Run the greeting pipeline and the recommendation job side by side.

    One background task, not two: ``BackgroundTasks`` awaits what it is
    given strictly in order, so a second entry would not start until the
    greeting's LLM calls had finished — and would never start at all if
    something escaped the first one. Gathered here they are genuinely
    concurrent, and neither can take the other down.
    """
    await asyncio.gather(
        _run_completion(user_id, recording_id, transcript, input_mode),
        _run_provider_recommendations(user_id, recording_id, transcript),
        return_exceptions=True,
    )


async def _run_completion(
    user_id: str,
    recording_id: str,
    transcript: str,
    input_mode: BrainDumpInputMode,
) -> None:
    """Background half of the pipeline, run after the response is sent.

    Anything that escapes here has no request to report to, so it marks
    the dump failed — that releases the preparing screen (it polls
    /status) instead of leaving it to hang until its ceiling.
    """
    try:
        await _extract_and_complete(user_id, recording_id, transcript, input_mode)
    except Exception as e:  # background task, nowhere to raise
        logger.error("Brain dump completion failed for user %s: %s", user_id, e)
        await db.mark_failed(user_id, recording_id, "understanding_failed")


async def _run_provider_recommendations(
    user_id: str, recording_id: str, transcript: str
) -> None:
    """The recommendation job, deliberately separate from ``_run_completion``.

    It writes only ``recommendedProviders`` — never the dump status — so a
    failure here can't strand the loading screen or mask a good greeting.
    ``generate_recommendations`` never raises; an empty result is stored
    as the final answer so the client stops polling.
    """
    recommendations = await recommend.generate_recommendations(transcript)
    try:
        await db.update_dump(
            user_id,
            recording_id,
            recommendedProviders=Json([r.model_dump() for r in recommendations]),
        )
    except Exception as e:  # background task, nowhere to raise
        logger.warning(
            "Brain dump provider recommendations not stored for user %s: %s",
            user_id,
            e,
        )


async def get_recommended_providers(user_id: str) -> RecommendedProvidersResponse:
    """Stored recommendations for the welcome dialog's connect panel.

    ``ready`` is true once the job has written its result (a null column
    means it is still running), or immediately when there is no transcript
    to recommend from. The dump ``status`` deliberately plays no part —
    it tracks the greeting pipeline, which can finish before this job.
    The client caps its own polling for rows the job never got to write
    (e.g. a process restart mid-job).
    """
    dump = await db.get_dump(user_id)
    if dump is None or not (dump.transcript or "").strip():
        return RecommendedProvidersResponse(ready=True)
    if dump.recommendedProviders is None:
        return RecommendedProvidersResponse(ready=False)
    return RecommendedProvidersResponse(
        ready=True, providers=_stored_recommendations(dump.recommendedProviders)
    )


def _stored_recommendations(raw: object) -> list[RecommendedProvider]:
    if not isinstance(raw, list):
        return []
    return [
        RecommendedProvider(provider=item["provider"], reason=item.get("reason", ""))
        for item in raw
        if isinstance(item, dict) and isinstance(item.get("provider"), str)
    ]


async def _extract_and_complete(
    user_id: str,
    recording_id: str,
    transcript: str,
    input_mode: BrainDumpInputMode,
) -> FinalizeResponse:
    # Losing this write means a newer take owns the row. The row writes
    # below would all no-op on their own, but the business understanding
    # is shared user context with no recording id on it — carrying on
    # would fold an abandoned take's transcript into the context the live
    # take is about to write. So the whole job stops here.
    if not await db.update_dump(
        user_id, recording_id, status=BrainDumpStatus.extracting
    ):
        return _superseded_response(user_id, input_mode)

    extracted = await _extract_understanding(user_id, transcript)
    extracted.additional_notes = _append_note(
        extracted.additional_notes, transcript, input_mode
    )
    # Re-checked rather than inferred from the claim above: extraction is
    # an LLM call, so the row has had seconds — minutes, for a long
    # transcript — to change hands since.
    if not await db.owns_dump(user_id, recording_id):
        return _superseded_response(user_id, input_mode)
    await upsert_business_understanding(user_id, extracted)

    # Generated here, while the onboarding loading screen is still up, so
    # the copilot home can render its greeting without waiting.
    greeting, suggested_prompts = await intro.generate_intro(transcript)

    await db.update_dump(
        user_id,
        recording_id,
        status=BrainDumpStatus.completed,
        errorCode=None,
        greeting=greeting,
        suggestedPrompts=Json([p.model_dump() for p in suggested_prompts]),
    )
    return _completed_response(transcript, input_mode)


async def _extract_understanding(
    user_id: str, transcript: str
) -> BusinessUnderstandingInput:
    understanding = await get_business_understanding(user_id)
    formatted = format_brain_dump_for_extraction(
        user_name=(understanding.user_name if understanding else None) or "",
        user_role=(understanding.user_role if understanding else None) or "",
        transcript=transcript,
    )
    try:
        return await extract_business_understanding(formatted)
    except Exception as e:
        # A failed extraction must not cost the user their transcript: the
        # raw text still lands in the understanding, so the copilot's
        # <user_context> is personalised even without structured fields.
        logger.warning("Brain dump extraction failed for user %s: %s", user_id, e)
        return BusinessUnderstandingInput.model_construct()


def _superseded_response(
    user_id: str, input_mode: BrainDumpInputMode
) -> FinalizeResponse:
    logger.info("Brain dump take for user %s superseded; stopping", user_id)
    return FinalizeResponse(
        status=BrainDumpStatus.failed, input_mode=input_mode, error_code="superseded"
    )


def _append_note(
    existing: str | None, transcript: str, input_mode: BrainDumpInputMode
) -> str:
    label = (
        "Onboarding brain dump (typed)"
        if input_mode == BrainDumpInputMode.typed
        else "Onboarding brain dump (spoken)"
    )
    note = f"{label}: {transcript}"
    return f"{existing}\n\n{note}" if existing else note


def _completed_response(
    transcript: str | None, input_mode: BrainDumpInputMode
) -> FinalizeResponse:
    return _pipeline_response(BrainDumpStatus.completed, transcript, input_mode)


def _pipeline_response(
    status: BrainDumpStatus,
    transcript: str | None,
    input_mode: BrainDumpInputMode,
) -> FinalizeResponse:
    return FinalizeResponse(
        status=status,
        input_mode=input_mode,
        transcript_preview=(transcript or "")[:TRANSCRIPT_PREVIEW_CHARS] or None,
    )


def _audio_filename(recording_id: str, mime_type: str | None) -> str:
    extensions = {
        "audio/webm": "webm",
        "audio/mp4": "m4a",
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/ogg": "ogg",
    }
    return f"brain-dump-{recording_id}.{extensions.get(mime_type or '', 'webm')}"


async def get_intro_card(user_id: str) -> IntroCardResponse:
    """The copilot home's greeting content.

    Path B covers every case where there is nothing to reflect back — the
    user skipped, or the pipeline failed and we have no transcript. A
    greeting that invites them to record is honest in both; one that
    claims to have heard them would not be.
    """
    dump = await db.get_dump(user_id)
    if dump is not None and dump.greetingSeen:
        # Content deliberately withheld: the client caches "done" and the
        # greeting must never reappear once the first message is sent.
        return IntroCardResponse(path="A", greeting="", greeting_done=True)

    if (
        dump is None
        or dump.inputMode == BrainDumpInputMode.skipped
        or not (dump.transcript or "").strip()
    ):
        return IntroCardResponse(
            path="B",
            greeting=prompts.PATH_B_GREETING,
            prompts=intro.fallback_prompts(),
        )

    greeting = (dump.greeting or "").strip()
    if not greeting and dump.status not in (
        BrainDumpStatus.completed,
        BrainDumpStatus.failed,
    ):
        # The background half of the pipeline is still writing the
        # greeting. An empty Path A response tells the client to keep
        # polling — serving the generic fallback to a brand-new user
        # would waste the personalised one that is seconds away.
        return IntroCardResponse(path="A", greeting="")
    if not greeting:
        # Completed before the greeting column existed, or generation
        # terminally failed after the transcript landed.
        greeting, _ = intro.fallback_intro(dump.transcript or "")
    return IntroCardResponse(
        path="A",
        greeting=greeting,
        prompts=_stored_prompts(dump.suggestedPrompts),
        transcript=dump.transcript,
    )


def _stored_prompts(raw: object) -> list[SuggestedPrompt]:
    """Rehydrate the stored prompt list, dropping anything malformed.

    A row written by an older build (or a hand-edited one) must degrade
    to the generic set rather than 500 the copilot home.
    """
    if not isinstance(raw, list):
        return intro.fallback_prompts()
    parsed = [
        SuggestedPrompt(
            title=item["title"],
            prompt=item["prompt"],
            icon=(
                item["icon"]
                if item.get("icon") in intro.PROMPT_ICONS
                else intro.DEFAULT_PROMPT_ICON
            ),
        )
        for item in raw
        if isinstance(item, dict)
        and isinstance(item.get("title"), str)
        and isinstance(item.get("prompt"), str)
    ]
    return parsed or intro.fallback_prompts()


async def mark_greeting_done(user_id: str) -> None:
    """Record that the user has completed the greeting (sent a message).

    Upserts so the flag also sticks for users who never produced a dump
    row — the greeting content, when any exists, is kept forever.
    """
    await db.mark_greeting_seen(user_id)
