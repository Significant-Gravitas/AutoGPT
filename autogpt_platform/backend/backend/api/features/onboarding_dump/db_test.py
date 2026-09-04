"""Guards on claiming the one-row-per-user brain dump."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma import Json
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import db

USER_ID = "user-1"
RECORDING_ID = "rec-1"


def _row(status: BrainDumpStatus, recording_id: str = RECORDING_ID):
    return OnboardingBrainDump.model_construct(
        userId=USER_ID,
        recordingId=recording_id,
        status=status,
        inputMode=BrainDumpInputMode.voice,
        transcript=None,
        errorCode=None,
    )


@pytest.fixture
def upsert(mocker: MockerFixture) -> AsyncMock:
    mock = AsyncMock(return_value=_row(BrainDumpStatus.recording_uploaded))
    mocker.patch.object(
        OnboardingBrainDump, "prisma", MagicMock(return_value=MagicMock(upsert=mock))
    )
    return mock


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        BrainDumpStatus.transcribing,
        BrainDumpStatus.transcribed,
        BrainDumpStatus.extracting,
        BrainDumpStatus.completed,
    ],
)
async def test_the_same_take_is_left_alone_once_it_is_moving(
    mocker: MockerFixture, upsert: AsyncMock, status: BrainDumpStatus
):
    """Recovery replays part 0, and finalize can be retried.

    Both land back here after the dump is already being processed. Left
    unguarded they reset the row to ``recording_uploaded`` underneath the
    running pipeline.
    """
    existing = _row(status)
    mocker.patch.object(db, "get_dump", AsyncMock(return_value=existing))

    result = await db.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.voice)

    assert result is existing
    assert result.status == status
    upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_take_is_reclaimed_so_retry_works(
    mocker: MockerFixture, upsert: AsyncMock
):
    mocker.patch.object(
        db, "get_dump", AsyncMock(return_value=_row(BrainDumpStatus.failed))
    )

    await db.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.voice)

    upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_new_recording_id_always_claims_the_row(
    mocker: MockerFixture, upsert: AsyncMock
):
    """Re-recording is a new take, even mid-pipeline on the old one."""
    mocker.patch.object(
        db,
        "get_dump",
        AsyncMock(return_value=_row(BrainDumpStatus.extracting, "rec-previous")),
    )

    await db.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.voice)

    upsert.assert_awaited_once()
    update = upsert.await_args.kwargs["data"]["update"]
    assert update["recordingId"] == RECORDING_ID
    # ``greetingSeen`` is only ever written True and short-circuits the
    # intro endpoint, so a new take that inherited it would run the whole
    # pipeline for a greeting that can never render.
    assert update["greetingSeen"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [BrainDumpStatus.recording_uploaded, BrainDumpStatus.completed],
)
async def test_a_delayed_typed_finalize_cannot_replace_an_active_take(
    mocker: MockerFixture, upsert: AsyncMock, status: BrainDumpStatus
):
    existing = _row(status, "rec-newer")
    mocker.patch.object(db, "get_dump", AsyncMock(return_value=existing))

    result = await db.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.typed)

    assert result is existing
    upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_new_recording_id_clears_every_column_the_last_take_owned(
    mocker: MockerFixture, upsert: AsyncMock
):
    """A new take must not be able to answer with the old take's work.

    There is one row per user, so anything left behind here is served as
    if it belonged to the new recording: ``/recording`` hands back the
    previous audio, ``/intro`` its greeting and transcript, and
    ``/recommended-providers`` its picks — for a take that may never get
    past its first part.
    """
    mocker.patch.object(
        db,
        "get_dump",
        AsyncMock(return_value=_row(BrainDumpStatus.completed, "rec-previous")),
    )

    await db.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.voice)

    update = upsert.await_args.kwargs["data"]["update"]
    assert update["audioPath"] is None
    assert update["mimeType"] is None
    assert update["sizeBytes"] is None
    assert update["durationSecs"] is None
    assert update["transcript"] is None
    assert update["transcriptLang"] is None
    assert update["greeting"] is None
    assert update["recommendedProviders"] == Json(None)
    # Non-nullable with a ``[]`` default, so it is emptied, not nulled.
    assert update["suggestedPrompts"] == Json([])


@pytest.mark.asyncio
async def test_retrying_the_same_take_keeps_what_that_take_produced(
    mocker: MockerFixture, upsert: AsyncMock
):
    """A retry is the same recording, not a replacement for it.

    Re-uploading part 0 after a failed finalize, or finalizing again, has
    to leave the audio and transcript that retry still needs.
    """
    mocker.patch.object(
        db, "get_dump", AsyncMock(return_value=_row(BrainDumpStatus.failed))
    )

    await db.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.voice)

    create = upsert.await_args.kwargs["data"]["create"]
    update = upsert.await_args.kwargs["data"]["update"]
    assert create["recommendedProviders"] == Json(None)
    assert set(update) == {"recordingId", "status", "inputMode", "errorCode"}


@pytest.mark.asyncio
async def test_greeting_only_create_sets_recommended_providers(
    upsert: AsyncMock,
):
    await db.mark_greeting_seen(USER_ID)

    create = upsert.await_args.kwargs["data"]["create"]
    assert create["recommendedProviders"] == Json(None)
    assert create["greetingSeen"] is True


@pytest.mark.asyncio
async def test_ownership_is_lost_as_soon_as_another_take_claims_the_row(
    mocker: MockerFixture,
):
    """Guards the one write that is not scoped to a take.

    The business understanding is shared user context, so the background
    job checks this before writing there.
    """
    mocker.patch.object(
        db,
        "get_dump",
        AsyncMock(return_value=_row(BrainDumpStatus.transcribing, "rec-2")),
    )
    assert await db.owns_dump(USER_ID, RECORDING_ID) is False

    mocker.patch.object(db, "get_dump", AsyncMock(return_value=None))
    assert await db.owns_dump(USER_ID, RECORDING_ID) is False

    mocker.patch.object(
        db, "get_dump", AsyncMock(return_value=_row(BrainDumpStatus.transcribing))
    )
    assert await db.owns_dump(USER_ID, RECORDING_ID) is True


@pytest.mark.asyncio
async def test_a_write_for_a_superseded_take_hits_nothing(mocker: MockerFixture):
    """Post-claim writes are scoped to the take that made them.

    One row per user means a second tab's take can own the row by the time
    a long transcription finishes. On ``userId`` alone the old take's
    transcript and greeting land on the new take's row.
    """
    update_many = AsyncMock(return_value=0)
    mocker.patch.object(
        OnboardingBrainDump,
        "prisma",
        MagicMock(return_value=MagicMock(update_many=update_many)),
    )

    updated = await db.update_dump(USER_ID, RECORDING_ID, transcript="anything")

    assert updated is False
    assert update_many.await_args.kwargs["where"] == {
        "userId": USER_ID,
        "recordingId": RECORDING_ID,
    }


@pytest.mark.asyncio
async def test_a_first_take_creates_the_row(mocker: MockerFixture, upsert: AsyncMock):
    mocker.patch.object(db, "get_dump", AsyncMock(return_value=None))

    await db.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.voice)

    upsert.assert_awaited_once()
