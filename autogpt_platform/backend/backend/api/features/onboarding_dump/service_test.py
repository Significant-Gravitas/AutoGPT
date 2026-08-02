"""Pipeline-level tests for the brain-dump service.

These call the service functions directly so the guarantees that matter
(the audio survives a failed transcription, the *complete* transcript is
what reaches the database and the understanding) can be asserted on the
stored row rather than inferred from a response body.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import intro, prompts, service
from backend.data.understanding import BusinessUnderstandingInput

USER_ID = "user-1"
RECORDING_ID = "rec-1"
TRANSCRIPT = "I run a bakery and want the weekly order emails handled."
AUDIO_PATH = "brain-dumps/rec-1.webm"
LONG_TRANSCRIPT = "the bakery ships pastries every friday morning " * 400


class DumpStore:
    """In-memory stand-in for the one-row-per-user brain dump table."""

    def __init__(self) -> None:
        self.row: OnboardingBrainDump | None = None
        self.statuses: list[BrainDumpStatus] = []
        self.transcripts: list[str | None] = []

    async def get_dump(self, user_id: str) -> OnboardingBrainDump | None:
        return self.row

    async def start_dump(
        self, user_id: str, recording_id: str, input_mode: BrainDumpInputMode
    ) -> OnboardingBrainDump:
        # model_construct skips defaults, so every column the code under
        # test reads has to be seeded explicitly — otherwise a plain
        # attribute access raises instead of returning the DB default.
        self.row = OnboardingBrainDump.model_construct(
            userId=user_id,
            recordingId=recording_id,
            status=BrainDumpStatus.recording_uploaded,
            inputMode=input_mode,
            transcript=None,
            greeting=None,
            suggestedPrompts=[],
            greetingSeen=False,
            audioPath=None,
            errorCode=None,
        )
        self.statuses.append(BrainDumpStatus.recording_uploaded)
        return self.row

    async def update_dump(
        self, user_id: str, **fields: Any
    ) -> OnboardingBrainDump | None:
        if self.row is None:
            return None
        for name, value in fields.items():
            setattr(self.row, name, value)
        status = fields.get("status")
        if status is not None:
            self.statuses.append(status)
        if "transcript" in fields:
            self.transcripts.append(fields["transcript"])
        return self.row

    async def mark_failed(self, user_id: str, error_code: str) -> None:
        await self.update_dump(
            user_id, status=BrainDumpStatus.failed, errorCode=error_code
        )


@pytest.fixture(autouse=True)
def dumps(mocker: MockerFixture) -> DumpStore:
    store = DumpStore()
    module = "backend.api.features.onboarding_dump.db"
    mocker.patch(f"{module}.get_dump", new=store.get_dump)
    mocker.patch(f"{module}.start_dump", new=store.start_dump)
    mocker.patch(f"{module}.update_dump", new=store.update_dump)
    mocker.patch(f"{module}.mark_failed", new=store.mark_failed)
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
    mock = AsyncMock(return_value=(TRANSCRIPT, "en"))
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
    transcribe.return_value = (LONG_TRANSCRIPT, "en")

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
    await dumps.update_dump(USER_ID, status=BrainDumpStatus.completed)

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
        USER_ID, status=BrainDumpStatus.failed, transcript="   ", errorCode="whatever"
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
        USER_ID, status=BrainDumpStatus.completed, transcript=TRANSCRIPT
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
