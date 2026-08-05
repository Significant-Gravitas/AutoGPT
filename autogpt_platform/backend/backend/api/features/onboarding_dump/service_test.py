"""Pipeline-level tests for the brain-dump service.

These call the service functions directly so the guarantees that matter
(the audio survives a failed transcription, the *complete* transcript is
what reaches the database and the understanding) can be asserted on the
stored row rather than inferred from a response body.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import (
    db,
    intro,
    prompts,
    service,
    transcription,
)
from backend.data.understanding import BusinessUnderstandingInput

USER_ID = "user-1"
RECORDING_ID = "rec-1"
TRANSCRIPT = "I run a bakery and want the weekly order emails handled."
AUDIO_PATH = "brain-dumps/rec-1.webm"
LONG_TRANSCRIPT = "the bakery ships pastries every friday morning " * 400

# The columns a new take clears, taken from the module under test so this
# fake cannot drift from it. A real read returns ``suggestedPrompts`` as a
# plain list rather than the ``Json`` wrapper used to write it.
_NEW_TAKE_COLUMNS: dict[str, Any] = {
    **db._TAKE_OWNED_RESET,
    "suggestedPrompts": [],
    "errorCode": None,
}


class DumpStore:
    """In-memory stand-in for the one-row-per-user brain dump table."""

    def __init__(self) -> None:
        self.row: OnboardingBrainDump | None = None
        self.statuses: list[BrainDumpStatus] = []
        self.transcripts: list[str | None] = []

    async def get_dump(self, user_id: str) -> OnboardingBrainDump | None:
        return self.row

    async def owns_dump(self, user_id: str, recording_id: str) -> bool:
        return self.row is not None and self.row.recordingId == recording_id

    async def start_dump(
        self, user_id: str, recording_id: str, input_mode: BrainDumpInputMode
    ) -> OnboardingBrainDump:
        # Mirrors the real `start_dump`: a take already moving through
        # the pipeline is returned untouched, so a replayed part 0 or a
        # repeated finalize cannot reset it.
        if (
            self.row is not None
            and self.row.status
            in (
                BrainDumpStatus.recording_uploaded,
                BrainDumpStatus.transcribing,
                BrainDumpStatus.transcribed,
                BrainDumpStatus.extracting,
                BrainDumpStatus.completed,
            )
            and (
                self.row.recordingId == recording_id
                or input_mode != BrainDumpInputMode.voice
            )
        ):
            return self.row
        if self.row is not None and self.row.recordingId == recording_id:
            # A retry of the same take: it keeps everything it has already
            # produced, exactly as the real claim does.
            self.row.status = BrainDumpStatus.recording_uploaded
            self.row.inputMode = input_mode
            self.row.errorCode = None
        else:
            # model_construct skips defaults, so every column the code
            # under test reads has to be seeded explicitly — otherwise a
            # plain attribute access raises instead of returning the DB
            # default. A different take starts from the cleared set.
            self.row = OnboardingBrainDump.model_construct(
                userId=user_id,
                recordingId=recording_id,
                status=BrainDumpStatus.recording_uploaded,
                inputMode=input_mode,
                **_NEW_TAKE_COLUMNS,
            )
        self.statuses.append(BrainDumpStatus.recording_uploaded)
        return self.row

    async def update_dump(self, user_id: str, recording_id: str, **fields: Any) -> bool:
        # Mirrors the scoped UPDATE: a write from a take the row has
        # already moved past hits nothing.
        if self.row is None or self.row.recordingId != recording_id:
            return False
        for name, value in fields.items():
            setattr(self.row, name, value)
        status = fields.get("status")
        if status is not None:
            self.statuses.append(status)
        if "transcript" in fields:
            self.transcripts.append(fields["transcript"])
        return True

    async def claim_transition(
        self,
        user_id: str,
        recording_id: str,
        *,
        expected: BrainDumpStatus,
        new: BrainDumpStatus,
        **fields: Any,
    ) -> bool:
        """Mirrors the conditional UPDATE: only matching rows transition."""
        if (
            self.row is None
            or self.row.recordingId != recording_id
            or self.row.status != expected
        ):
            return False
        await self.update_dump(user_id, recording_id, status=new, **fields)
        return True

    async def mark_failed(
        self, user_id: str, recording_id: str, error_code: str
    ) -> None:
        await self.update_dump(
            user_id,
            recording_id,
            status=BrainDumpStatus.failed,
            errorCode=error_code,
        )


@pytest.fixture(autouse=True)
def dumps(mocker: MockerFixture) -> DumpStore:
    store = DumpStore()
    module = "backend.api.features.onboarding_dump.db"
    mocker.patch(f"{module}.get_dump", new=store.get_dump)
    mocker.patch(f"{module}.owns_dump", new=store.owns_dump)
    mocker.patch(f"{module}.start_dump", new=store.start_dump)
    mocker.patch(f"{module}.update_dump", new=store.update_dump)
    mocker.patch(f"{module}.mark_failed", new=store.mark_failed)
    mocker.patch(f"{module}.claim_transition", new=store.claim_transition)
    return store


@pytest.fixture(autouse=True)
def storage_mocks(mocker: MockerFixture) -> dict[str, AsyncMock]:
    module = "backend.api.features.onboarding_dump.storage"
    mocks = {
        "assemble_parts": AsyncMock(return_value=b"opus-audio-bytes"),
        "store_audio": AsyncMock(return_value=AUDIO_PATH),
        "discard_parts": AsyncMock(),
    }
    for name, mock in mocks.items():
        mocker.patch(f"{module}.{name}", new=mock)
    return mocks


@pytest.fixture(autouse=True)
def transcribe(mocker: MockerFixture) -> AsyncMock:
    mock = AsyncMock(
        return_value=transcription.TranscriptionResult(
            text=TRANSCRIPT, language="en", model="gpt-4o-transcribe"
        )
    )
    mocker.patch(
        "backend.api.features.onboarding_dump.transcription.transcribe", new=mock
    )
    return mock


@pytest.fixture(autouse=True)
def extraction(mocker: MockerFixture) -> dict[str, AsyncMock]:
    module = "backend.api.features.onboarding_dump.service"
    mocks = {
        "scan_content_safe": AsyncMock(),
        "get_business_understanding": AsyncMock(return_value=None),
        "extract_business_understanding": AsyncMock(
            return_value=BusinessUnderstandingInput.model_construct()
        ),
        "upsert_business_understanding": AsyncMock(),
    }
    for name, mock in mocks.items():
        mocker.patch(f"{module}.{name}", new=mock)
    return mocks


async def start_voice_take(dumps: DumpStore) -> None:
    await dumps.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.voice)


async def finalize_voice(duration_secs: float = 12.0, mime_type: str | None = None):
    """Run finalize plus its background half (extraction + greeting),
    the way starlette does after the response is sent."""
    background = BackgroundTasks()
    response = await service.finalize_voice_dump(
        USER_ID, RECORDING_ID, duration_secs, mime_type, background
    )
    await background()
    return response


@pytest.mark.asyncio
async def test_transcription_failure_keeps_the_audio_and_records_the_error(
    dumps: DumpStore, transcribe: AsyncMock
):
    await start_voice_take(dumps)
    transcribe.side_effect = RuntimeError("provider blew up")

    response = await finalize_voice()

    assert response.status == BrainDumpStatus.failed
    assert response.error_code == "transcription_failed"
    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.failed
    assert dumps.row.errorCode == "transcription_failed"
    # The zero-lost-recordings guarantee: the audio is still addressable.
    assert dumps.row.audioPath == AUDIO_PATH


@pytest.mark.asyncio
async def test_missing_stt_provider_is_reported_as_unavailable(
    dumps: DumpStore, transcribe: AsyncMock
):
    await start_voice_take(dumps)
    transcribe.side_effect = service.transcription.TranscriptionUnavailableError(
        "no key"
    )

    response = await finalize_voice()

    assert response.error_code == "transcription_unavailable"
    assert dumps.row is not None
    assert dumps.row.errorCode == "transcription_unavailable"
    assert dumps.row.audioPath == AUDIO_PATH


@pytest.mark.asyncio
async def test_audio_storage_failure_is_retryable(
    dumps: DumpStore,
    storage_mocks: dict[str, AsyncMock],
    transcribe: AsyncMock,
):
    await start_voice_take(dumps)
    storage_mocks["store_audio"].side_effect = RuntimeError("storage unavailable")

    response = await finalize_voice()

    assert response.status == BrainDumpStatus.failed
    assert response.error_code == "storage_failed"
    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.failed
    assert dumps.row.errorCode == "storage_failed"
    storage_mocks["discard_parts"].assert_not_awaited()
    transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_transcript_lands_in_the_business_understanding(
    dumps: DumpStore, extraction: dict[str, AsyncMock]
):
    await start_voice_take(dumps)

    await finalize_voice(mime_type="audio/webm")

    upsert = extraction["upsert_business_understanding"]
    upsert.assert_awaited_once()
    user_id, understanding = upsert.await_args.args
    assert user_id == USER_ID
    assert isinstance(understanding, BusinessUnderstandingInput)
    assert understanding.additional_notes == (
        f"Onboarding brain dump (spoken): {TRANSCRIPT}"
    )


@pytest.mark.asyncio
async def test_typed_dump_is_labelled_as_typed_in_the_understanding(
    extraction: dict[str, AsyncMock]
):
    background = BackgroundTasks()
    await service.finalize_typed_dump(
        USER_ID, RECORDING_ID, "I run a bakery.", background
    )
    await background()

    _, understanding = extraction["upsert_business_understanding"].await_args.args
    assert understanding.additional_notes == (
        "Onboarding brain dump (typed): I run a bakery."
    )


@pytest.mark.asyncio
async def test_repeating_a_typed_finalize_does_not_restart_the_pipeline(
    dumps: DumpStore, extraction: dict[str, AsyncMock]
):
    """A second submit of the same typed take is a no-op.

    Without the guard the row was re-claimed — dropping an in-flight
    extraction back to ``recording_uploaded`` — and a second pair of
    background jobs was queued on top of the running ones.
    """
    first = BackgroundTasks()
    await service.finalize_typed_dump(USER_ID, RECORDING_ID, "I run a bakery.", first)
    await first()
    statuses_after_first = list(dumps.statuses)

    second = BackgroundTasks()
    response = await service.finalize_typed_dump(
        USER_ID, RECORDING_ID, "I run a bakery.", second
    )

    assert response.status == BrainDumpStatus.completed
    assert dumps.statuses == statuses_after_first
    assert BrainDumpStatus.recording_uploaded not in statuses_after_first[1:]
    # Nothing queued the second time round.
    assert second.tasks == []
    assert extraction["upsert_business_understanding"].await_count == 1


@pytest.mark.asyncio
async def test_a_stale_typed_finalize_cannot_replace_a_completed_take(
    dumps: DumpStore,
):
    newer_tasks = BackgroundTasks()
    await service.finalize_typed_dump(
        USER_ID, "rec-newer", "This is the newer transcript.", newer_tasks
    )
    await newer_tasks()
    assert dumps.row is not None
    newer_transcript = dumps.row.transcript
    stale_tasks = BackgroundTasks()

    response = await service.finalize_typed_dump(
        USER_ID, "rec-stale", "This stale request arrived late.", stale_tasks
    )

    assert response.status == BrainDumpStatus.failed
    assert response.error_code == "superseded"
    assert dumps.row.recordingId == "rec-newer"
    assert dumps.row.transcript == newer_transcript
    assert stale_tasks.tasks == []


@pytest.mark.asyncio
async def test_typed_finalize_cannot_replace_an_uploaded_voice_take(
    dumps: DumpStore,
):
    await start_voice_take(dumps)
    tasks = BackgroundTasks()

    response = await service.finalize_typed_dump(
        USER_ID, "rec-typed", "Typed text arrived from another tab.", tasks
    )

    assert response.status == BrainDumpStatus.failed
    assert response.error_code == "superseded"
    assert dumps.row is not None
    assert dumps.row.recordingId == RECORDING_ID
    assert dumps.row.status == BrainDumpStatus.recording_uploaded
    assert dumps.row.inputMode == BrainDumpInputMode.voice
    assert tasks.tasks == []


@pytest.mark.asyncio
async def test_typed_finalize_does_not_relabel_voice_transcription(
    dumps: DumpStore,
):
    await start_voice_take(dumps)
    await dumps.update_dump(USER_ID, RECORDING_ID, status=BrainDumpStatus.transcribing)
    tasks = BackgroundTasks()

    response = await service.finalize_typed_dump(
        USER_ID, RECORDING_ID, "Typed text arrived late.", tasks
    )

    assert response.status == BrainDumpStatus.transcribing
    assert response.input_mode == BrainDumpInputMode.voice
    assert dumps.row is not None
    assert dumps.row.inputMode == BrainDumpInputMode.voice
    assert tasks.tasks == []


def release_both_past_the_guard(mocker: MockerFixture, dumps: "DumpStore"):
    """Hold the first two `get_dump` calls until both have happened.

    Without this the coroutines never interleave — none of the fakes
    suspend, so `asyncio.gather` just runs one to completion and then the
    other, and the race under test never occurs.
    """
    barrier = asyncio.Barrier(2)
    inner = dumps.get_dump
    arrivals = 0

    async def gated(user_id: str):
        nonlocal arrivals
        arrivals += 1
        row = await inner(user_id)
        # Snapshot first, *then* wait. The read is what races: both
        # callers must come away holding the status as it was before
        # either of them acted. The store mutates its single row in
        # place, so without the copy the loser would silently observe
        # the winner's later writes and the race would vanish.
        snapshot = row.model_copy() if row is not None else None
        if arrivals <= 2:
            await barrier.wait()
        return snapshot

    mocker.patch("backend.api.features.onboarding_dump.db.get_dump", new=gated)


@pytest.mark.asyncio
async def test_two_concurrent_voice_finalizes_only_process_the_take_once(
    mocker: MockerFixture,
    dumps: DumpStore,
    storage_mocks: dict[str, AsyncMock],
    transcribe: AsyncMock,
):
    """Only the caller that wins the atomic claim does the work.

    Both requests read the guard while the status is still
    `recording_uploaded`, so both pass it. Without the conditional UPDATE
    both then assembled, stored and transcribed the same recording.
    """
    await start_voice_take(dumps)
    release_both_past_the_guard(mocker, dumps)

    await asyncio.gather(
        service.finalize_voice_dump(
            USER_ID, RECORDING_ID, 12.0, None, BackgroundTasks()
        ),
        service.finalize_voice_dump(
            USER_ID, RECORDING_ID, 12.0, None, BackgroundTasks()
        ),
    )

    transcribe.assert_awaited_once()
    storage_mocks["store_audio"].assert_awaited_once()


@pytest.mark.asyncio
async def test_two_concurrent_typed_finalizes_only_queue_one_pipeline(
    mocker: MockerFixture, dumps: DumpStore, extraction: dict[str, AsyncMock]
):
    await dumps.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.typed)
    release_both_past_the_guard(mocker, dumps)
    first_tasks = BackgroundTasks()
    second_tasks = BackgroundTasks()

    await asyncio.gather(
        service.finalize_typed_dump(
            USER_ID, RECORDING_ID, "I run a bakery.", first_tasks
        ),
        service.finalize_typed_dump(
            USER_ID, RECORDING_ID, "I run a bakery.", second_tasks
        ),
    )
    await first_tasks()
    await second_tasks()

    # One winner queues the extraction/greeting pair; the loser queues
    # nothing, so the understanding is written exactly once.
    assert extraction["upsert_business_understanding"].await_count == 1


@pytest.mark.asyncio
async def test_a_typed_finalize_that_loses_the_claim_queues_nothing(
    mocker: MockerFixture, dumps: DumpStore, extraction: dict[str, AsyncMock]
):
    """Losing the claim must not fall through to the background jobs.

    The earlier version returned early only when the row still belonged
    to this take; any other losing case carried on and queued a second
    extraction and greeting on top of the winner's.
    """
    await dumps.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.typed)
    mocker.patch(
        "backend.api.features.onboarding_dump.db.claim_transition",
        new=AsyncMock(return_value=False),
    )
    tasks = BackgroundTasks()

    response = await service.finalize_typed_dump(
        USER_ID, RECORDING_ID, "I run a bakery.", tasks
    )

    assert tasks.tasks == []
    assert response.input_mode == BrainDumpInputMode.typed
    await tasks()
    extraction["upsert_business_understanding"].assert_not_awaited()


@pytest.mark.asyncio
async def test_a_voice_finalize_that_loses_the_claim_does_no_work(
    mocker: MockerFixture,
    dumps: DumpStore,
    storage_mocks: dict[str, AsyncMock],
    transcribe: AsyncMock,
):
    await start_voice_take(dumps)
    mocker.patch(
        "backend.api.features.onboarding_dump.db.claim_transition",
        new=AsyncMock(return_value=False),
    )

    await service.finalize_voice_dump(
        USER_ID, RECORDING_ID, 12.0, None, BackgroundTasks()
    )

    storage_mocks["assemble_parts"].assert_not_awaited()
    storage_mocks["store_audio"].assert_not_awaited()
    transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_voice_finalize_for_a_superseded_take_reports_its_own_take(
    dumps: DumpStore, storage_mocks: dict[str, AsyncMock], transcribe: AsyncMock
):
    """A second tab's take owns the row — say so, don't mirror it.

    There is one row per user, so a newer recording overwrites the older
    one. The losing finalize used to return whatever status it found,
    which is the *other* take's progress dressed up as this one: the old
    tab would narrate a recording it never made, and could be walked
    through to a greeting built from audio the user recorded elsewhere.
    """
    await start_voice_take(dumps)
    # Second tab starts its own take; the row is now theirs.
    await dumps.start_dump(USER_ID, "rec-2", BrainDumpInputMode.voice)

    response = await service.finalize_voice_dump(
        USER_ID, RECORDING_ID, 12.0, None, BackgroundTasks()
    )

    assert response.status == BrainDumpStatus.failed
    assert response.error_code == "superseded"
    # The live take must come through untouched — no status write, and
    # none of the pipeline run against the wrong recording.
    assert dumps.row is not None
    assert dumps.row.recordingId == "rec-2"
    assert dumps.row.status == BrainDumpStatus.recording_uploaded
    storage_mocks["assemble_parts"].assert_not_awaited()
    transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_take_superseded_mid_transcription_never_writes_to_the_new_row(
    dumps: DumpStore, transcribe: AsyncMock
):
    """The old take loses the row the moment a second tab claims it.

    Transcription can run for minutes. Scoped on ``userId`` alone, every
    write after the claim — transcript, status, greeting, failure code —
    landed on whatever take held the row by the time it finished. The new
    recording then looked already-transcribed to the idempotency guard
    and was never transcribed at all.
    """
    await start_voice_take(dumps)

    async def supersede(*args: Any, **kwargs: Any):
        await dumps.start_dump(USER_ID, "rec-2", BrainDumpInputMode.voice)
        return transcription.TranscriptionResult(
            text=TRANSCRIPT, language="en", model="gpt-4o-transcribe"
        )

    transcribe.side_effect = supersede

    await finalize_voice()

    assert dumps.row is not None
    assert dumps.row.recordingId == "rec-2"
    assert dumps.row.transcript is None
    assert dumps.row.greeting is None
    assert dumps.row.status == BrainDumpStatus.recording_uploaded


@pytest.mark.asyncio
async def test_a_superseded_take_never_writes_to_the_shared_understanding(
    dumps: DumpStore, extraction: dict[str, AsyncMock]
):
    """The background half has to stop, not just fail to write its row.

    Row writes carry a ``recordingId`` guard so they no-op once a second
    tab owns the row, but the business understanding is shared user
    context with no take on it. An abandoned take that kept going folded
    its transcript into the context the live take is about to write.
    """
    await start_voice_take(dumps)
    background = BackgroundTasks()
    await service.finalize_voice_dump(USER_ID, RECORDING_ID, 12.0, None, background)
    # A second tab claims the row between the response and the background
    # half that starlette runs after it.
    await dumps.start_dump(USER_ID, "rec-2", BrainDumpInputMode.voice)

    await background()

    extraction["upsert_business_understanding"].assert_not_awaited()
    assert dumps.row is not None
    assert dumps.row.recordingId == "rec-2"
    assert dumps.row.status == BrainDumpStatus.recording_uploaded


@pytest.mark.asyncio
async def test_a_take_superseded_during_extraction_stops_before_the_understanding(
    dumps: DumpStore, extraction: dict[str, AsyncMock]
):
    """Ownership is re-checked next to the understanding write.

    Extraction is an LLM call, so passing the claim before it says
    nothing about who owns the row after it.
    """
    await start_voice_take(dumps)

    async def supersede(*args: Any, **kwargs: Any) -> BusinessUnderstandingInput:
        await dumps.start_dump(USER_ID, "rec-2", BrainDumpInputMode.voice)
        return BusinessUnderstandingInput.model_construct()

    extraction["extract_business_understanding"].side_effect = supersede

    await finalize_voice()

    extraction["upsert_business_understanding"].assert_not_awaited()


@pytest.mark.asyncio
async def test_a_new_take_never_serves_the_previous_takes_answers(dumps: DumpStore):
    """One row per user, so a new take inherits the last one's columns.

    Left there, the user re-records, gets nowhere, and the copilot greets
    them with the greeting and transcript of a take they replaced —
    while ``/recording`` hands back its audio.
    """
    await start_voice_take(dumps)
    await finalize_voice()
    await dumps.update_dump(USER_ID, RECORDING_ID, greetingSeen=True)
    assert dumps.row is not None and dumps.row.greeting

    # They record again and never finish this one.
    await dumps.start_dump(USER_ID, "rec-2", BrainDumpInputMode.voice)

    card = await service.get_intro_card(USER_ID)
    providers = await service.get_recommended_providers(USER_ID)

    assert dumps.row.transcript is None
    assert dumps.row.greeting is None
    assert dumps.row.audioPath is None
    assert card.path == "B"
    assert card.greeting == prompts.PATH_B_GREETING
    assert card.transcript is None
    assert card.greeting_done is False
    assert providers.ready is True
    assert providers.providers == []


@pytest.mark.asyncio
async def test_a_failed_new_take_gets_an_answer_rather_than_an_endless_wait(
    dumps: DumpStore, transcribe: AsyncMock
):
    """Claiming a new take clears ``greetingSeen``, so the intro is live again.

    That is right while a new greeting is on its way, and it must not
    strand a user whose new take dies: an empty Path A tells the client
    to keep polling, so a failed take has to resolve to Path B instead.
    """
    await start_voice_take(dumps)
    await finalize_voice()
    await dumps.update_dump(USER_ID, RECORDING_ID, greetingSeen=True)

    await dumps.start_dump(USER_ID, "rec-2", BrainDumpInputMode.voice)
    transcribe.side_effect = RuntimeError("provider blew up")
    background = BackgroundTasks()
    await service.finalize_voice_dump(USER_ID, "rec-2", 12.0, None, background)
    await background()

    card = await service.get_intro_card(USER_ID)

    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.failed
    assert card.path == "B"
    assert card.greeting == prompts.PATH_B_GREETING


@pytest.mark.asyncio
async def test_finalizing_a_take_that_never_uploaded_writes_no_row(
    dumps: DumpStore, storage_mocks: dict[str, AsyncMock], transcribe: AsyncMock
):
    """Finalize without a single part told the caller, not the database.

    Nothing was ever uploaded, so there is no row to fail — and creating
    one here would let a late finalize resurrect a dump for a user whose
    row was deleted, which ``update_dump`` exists to prevent. The browser
    still holds every part in IndexedDB, so the caller re-uploads on the
    strength of the response alone.
    """
    storage_mocks["assemble_parts"].return_value = b""

    response = await service.finalize_voice_dump(
        USER_ID, RECORDING_ID, 12.0, None, BackgroundTasks()
    )

    assert response.status == BrainDumpStatus.failed
    assert response.error_code == "no_audio_received"
    assert dumps.row is None
    transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_buffer_with_no_row_is_kept_rather_than_processed_into_nothing(
    dumps: DumpStore, storage_mocks: dict[str, AsyncMock], transcribe: AsyncMock
):
    """Parts but no row is the one shape that could lose a recording.

    Every ``update_dump`` keys on ``userId`` and silently no-ops when
    there is no row, so processing here would store the audio, drop the
    Redis buffer and hand the client a success it can never read back —
    the transcript, the greeting and the status all written nowhere. The
    buffer has to survive so a re-upload (part 0 recreates the row) can
    still recover the take.
    """
    storage_mocks["assemble_parts"].return_value = b"opus-audio-bytes"

    response = await service.finalize_voice_dump(
        USER_ID, RECORDING_ID, 12.0, None, BackgroundTasks()
    )

    assert response.status == BrainDumpStatus.failed
    assert response.error_code == "no_audio_received"
    assert dumps.row is None
    storage_mocks["store_audio"].assert_not_awaited()
    storage_mocks["discard_parts"].assert_not_awaited()
    transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_extraction_still_preserves_the_transcript(
    dumps: DumpStore, extraction: dict[str, AsyncMock]
):
    await start_voice_take(dumps)
    extraction["extract_business_understanding"].side_effect = RuntimeError("no llm")

    response = await finalize_voice()

    # Finalize answers as soon as the transcript is stored; the failed
    # extraction is absorbed by the background half.
    assert response.status == BrainDumpStatus.transcribed
    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.completed
    _, understanding = extraction["upsert_business_understanding"].await_args.args
    assert TRANSCRIPT in (understanding.additional_notes or "")


@pytest.mark.asyncio
async def test_long_transcript_is_stored_whole(
    dumps: DumpStore, transcribe: AsyncMock, extraction: dict[str, AsyncMock]
):
    await start_voice_take(dumps)
    transcribe.return_value = transcription.TranscriptionResult(
        text=LONG_TRANSCRIPT, language="en", model="gpt-4o-transcribe"
    )

    await finalize_voice(duration_secs=900.0)

    assert dumps.transcripts == [LONG_TRANSCRIPT]
    assert dumps.row is not None
    assert dumps.row.transcript == LONG_TRANSCRIPT
    _, understanding = extraction["upsert_business_understanding"].await_args.args
    assert LONG_TRANSCRIPT in (understanding.additional_notes or "")


@pytest.mark.asyncio
async def test_intro_card_takes_path_a_with_the_stored_greeting(dumps: DumpStore):
    await dumps.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.voice)
    await dumps.update_dump(
        USER_ID,
        RECORDING_ID,
        status=BrainDumpStatus.completed,
        transcript=TRANSCRIPT,
        greeting="You mentioned the weekly order emails.",
        suggestedPrompts=[
            {"title": "Handle the weekly order emails", "prompt": "Please handle them."}
        ],
    )

    card = await service.get_intro_card(USER_ID)

    assert card.path == "A"
    assert card.greeting == "You mentioned the weekly order emails."
    assert [p.title for p in card.prompts] == ["Handle the weekly order emails"]
    assert [p.prompt for p in card.prompts] == ["Please handle them."]
    assert card.greeting_done is False
    assert card.transcript == TRANSCRIPT


@pytest.mark.asyncio
async def test_intro_card_takes_path_b_when_the_user_skipped(dumps: DumpStore):
    await dumps.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.skipped)
    await dumps.update_dump(USER_ID, RECORDING_ID, status=BrainDumpStatus.completed)

    card = await service.get_intro_card(USER_ID)

    assert card.path == "B"
    assert card.greeting == prompts.PATH_B_GREETING
    assert card.prompts == intro.fallback_prompts()
    assert card.greeting_done is False
    assert card.transcript is None


@pytest.mark.asyncio
async def test_intro_card_takes_path_b_when_the_transcript_is_empty(dumps: DumpStore):
    await dumps.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.voice)
    await dumps.update_dump(
        USER_ID,
        RECORDING_ID,
        status=BrainDumpStatus.failed,
        transcript="   ",
        errorCode="whatever",
    )

    card = await service.get_intro_card(USER_ID)

    assert card.path == "B"
    assert card.greeting == prompts.PATH_B_GREETING


@pytest.mark.asyncio
async def test_intro_card_takes_path_b_without_a_dump_row():
    card = await service.get_intro_card(USER_ID)

    assert card.path == "B"
    assert card.greeting == prompts.PATH_B_GREETING
    assert card.prompts == intro.fallback_prompts()
    assert card.greeting_done is False


@pytest.mark.asyncio
async def test_intro_card_is_withheld_once_the_greeting_was_seen(dumps: DumpStore):
    await dumps.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.voice)
    await dumps.update_dump(
        USER_ID,
        RECORDING_ID,
        status=BrainDumpStatus.completed,
        transcript=TRANSCRIPT,
        greeting="You mentioned the weekly order emails.",
        greetingSeen=True,
    )

    card = await service.get_intro_card(USER_ID)

    assert card.greeting_done is True
    assert card.greeting == ""
    assert card.prompts == []
    assert card.transcript is None


@pytest.mark.asyncio
async def test_intro_card_falls_back_when_the_greeting_was_never_generated(
    dumps: DumpStore,
):
    # A dump processed before the greeting existed, or one whose generation
    # failed. Still Path A — we did hear them — just without the prose.
    await dumps.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.voice)
    await dumps.update_dump(
        USER_ID, RECORDING_ID, status=BrainDumpStatus.completed, transcript=TRANSCRIPT
    )

    card = await service.get_intro_card(USER_ID)

    assert card.path == "A"
    assert card.greeting == intro.fallback_intro(TRANSCRIPT)[0]
    # An empty stored list degrades to the generic starters — a greeting
    # with nothing under it would look broken.
    assert card.prompts == intro.fallback_prompts()


@pytest.mark.asyncio
async def test_intro_generation_degrades_instead_of_raising(mocker: MockerFixture):
    # The greeting must never be the reason onboarding fails, so a broken
    # LLM call resolves to the template rather than propagating.
    mocker.patch(
        "backend.api.features.onboarding_dump.intro.get_openai_client",
        return_value=None,
    )

    greeting, suggested = await intro.generate_intro(TRANSCRIPT)

    assert greeting == intro.fallback_intro(TRANSCRIPT)[0]
    assert suggested == intro.fallback_prompts()
