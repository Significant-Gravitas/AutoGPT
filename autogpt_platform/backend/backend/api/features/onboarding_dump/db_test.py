"""Guards on claiming the one-row-per-user brain dump."""

from unittest.mock import AsyncMock, MagicMock

import pytest
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
    assert upsert.await_args.kwargs["data"]["update"]["recordingId"] == RECORDING_ID


@pytest.mark.asyncio
async def test_a_first_take_creates_the_row(mocker: MockerFixture, upsert: AsyncMock):
    mocker.patch.object(db, "get_dump", AsyncMock(return_value=None))

    await db.start_dump(USER_ID, RECORDING_ID, BrainDumpInputMode.voice)

    upsert.assert_awaited_once()
