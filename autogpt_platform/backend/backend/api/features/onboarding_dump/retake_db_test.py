"""A new take after a finished one, against Postgres.

``service_test.py`` runs the service on an in-memory copy of ``start_dump``'s
rules, so only the database shows which takes the real claim lets through.
"""

import io
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import BackgroundTasks, HTTPException, UploadFile
from prisma import Json
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import User
from pytest_mock import MockerFixture
from starlette.datastructures import Headers

from backend.api.features.onboarding_dump import db, routes, service

TYPED_TEXT = "I run a bakery and spend Mondays reconciling supplier invoices."


@pytest_asyncio.fixture(loop_scope="session")
async def user_id(server) -> AsyncGenerator[str, None]:
    user_id = str(uuid.uuid4())
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    yield user_id
    await User.prisma().delete(where={"id": user_id})


@pytest.fixture(autouse=True)
def admitted(mocker: MockerFixture) -> None:
    mocker.patch.object(service, "enforce_personalization_budget", new=AsyncMock())
    mocker.patch(
        "backend.api.features.onboarding_dump.quality.check_transcript_quality",
        new=AsyncMock(return_value=None),
    )


async def test_a_typed_take_after_a_skip_is_processed(user_id: str):
    await service.finalize_skipped_dump(user_id, "rec-skip")
    tasks = BackgroundTasks()

    response = await service.finalize_typed_dump(
        user_id, "rec-typed", TYPED_TEXT, tasks
    )

    assert response.status == BrainDumpStatus.transcribed
    assert len(tasks.tasks) == 1
    dump = await db.get_dump(user_id)
    assert dump is not None
    assert (dump.recordingId, dump.inputMode, dump.transcript) == (
        "rec-typed",
        BrainDumpInputMode.typed,
        TYPED_TEXT,
    )

    repeat = BackgroundTasks()
    again = await service.finalize_typed_dump(user_id, "rec-typed", TYPED_TEXT, repeat)
    assert again.status == BrainDumpStatus.transcribed
    assert repeat.tasks == []


async def test_a_skip_after_a_completed_take_replaces_it(user_id: str):
    await _completed_typed_take(user_id, "rec-typed")

    await service.finalize_skipped_dump(user_id, "rec-skip")

    dump = await db.get_dump(user_id)
    assert dump is not None
    assert (dump.recordingId, dump.status, dump.inputMode) == (
        "rec-skip",
        BrainDumpStatus.completed,
        BrainDumpInputMode.skipped,
    )
    assert dump.transcript is None
    assert dump.greeting is None
    assert dump.recommendedExperts is None


async def test_a_skip_after_an_abandoned_upload_lets_the_next_take_through(
    user_id: str, mocker: MockerFixture
):
    """Part 0 claimed the row, a later part never arrived, and the user skipped."""
    mocker.patch.object(routes.storage, "discard_parts", new=AsyncMock())
    await db.start_dump(user_id, "rec-voice", BrainDumpInputMode.voice)

    await routes.discard_brain_dump(user_id=user_id, recording_id="rec-voice")
    await service.finalize_skipped_dump(user_id, "rec-skip")

    skipped = await db.get_dump(user_id)
    assert skipped is not None
    assert (skipped.recordingId, skipped.inputMode) == (
        "rec-skip",
        BrainDumpInputMode.skipped,
    )
    response = await service.finalize_typed_dump(
        user_id, "rec-typed", TYPED_TEXT, BackgroundTasks()
    )
    assert response.status == BrainDumpStatus.transcribed


async def test_an_abandoned_upload_is_closed_even_when_its_buffer_cannot_be_dropped(
    user_id: str, mocker: MockerFixture
):
    mocker.patch.object(
        routes.storage, "discard_parts", new=AsyncMock(side_effect=OSError("redis"))
    )
    await db.start_dump(user_id, "rec-voice", BrainDumpInputMode.voice)

    with pytest.raises(OSError):
        await routes.discard_brain_dump(user_id=user_id, recording_id="rec-voice")

    dump = await db.get_dump(user_id)
    assert dump is not None
    assert (dump.status, dump.errorCode) == (BrainDumpStatus.failed, "discarded")


async def test_a_late_first_part_of_a_skipped_take_cannot_reopen_it(user_id: str):
    """Part 0 was still on the wire when the user skipped, and lands afterwards."""
    await routes.discard_brain_dump(user_id=user_id, recording_id="rec-voice")
    await service.finalize_skipped_dump(user_id, "rec-skip")

    with pytest.raises(HTTPException) as refused:
        await _upload_part_zero(user_id, "rec-voice")

    assert refused.value.status_code == 409
    dump = await db.get_dump(user_id)
    assert dump is not None
    assert (dump.recordingId, dump.inputMode) == (
        "rec-skip",
        BrainDumpInputMode.skipped,
    )


async def test_a_skipped_take_resent_from_the_browser_starts_again(user_id: str):
    """Back from Hire, then "Use that recording", sends the discarded take on purpose."""
    await routes.discard_brain_dump(user_id=user_id, recording_id="rec-voice")
    await service.finalize_skipped_dump(user_id, "rec-skip")

    await _upload_part_zero(user_id, "rec-voice", resent=True)

    dump = await db.get_dump(user_id)
    assert dump is not None
    assert (dump.recordingId, dump.status) == (
        "rec-voice",
        BrainDumpStatus.recording_uploaded,
    )


# Another tab's take may be the one running, so neither kind of new take
# can reset it.
@pytest.mark.parametrize("second", ["typed", "skipped"])
async def test_a_new_take_cannot_displace_one_still_being_processed(
    user_id: str, second: str
):
    await service.finalize_typed_dump(
        user_id, "rec-first", TYPED_TEXT, BackgroundTasks()
    )
    tasks = BackgroundTasks()

    if second == "typed":
        response = await service.finalize_typed_dump(
            user_id, "rec-second", "Something else entirely.", tasks
        )
        assert response.error_code == "superseded"
    else:
        await service.finalize_skipped_dump(user_id, "rec-second")

    assert tasks.tasks == []
    dump = await db.get_dump(user_id)
    assert dump is not None
    assert (dump.recordingId, dump.status, dump.transcript) == (
        "rec-first",
        BrainDumpStatus.transcribed,
        TYPED_TEXT,
    )


async def _completed_typed_take(user_id: str, recording_id: str) -> None:
    await db.start_dump(user_id, recording_id, BrainDumpInputMode.typed)
    await db.update_dump(
        user_id,
        recording_id,
        status=BrainDumpStatus.completed,
        transcript=TYPED_TEXT,
        greeting="You run a bakery.",
        recommendedExperts=Json({"experts": []}),
    )


async def _upload_part_zero(user_id: str, recording_id: str, resent: bool = False):
    part = UploadFile(
        file=io.BytesIO(b"opus"), headers=Headers({"content-type": "audio/webm"})
    )
    return await routes.upload_brain_dump_part(
        user_id=user_id,
        file=part,
        recording_id=recording_id,
        part_index=0,
        resent=resent,
    )
