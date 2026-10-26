"""A new take after a finished one, against Postgres.

``service_test.py`` runs the service on an in-memory copy of ``start_dump``'s
rules, so only the database shows which takes the real claim lets through.
"""

import asyncio
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


async def test_a_late_first_part_claiming_the_row_after_the_discard_gives_it_back(
    user_id: str, mocker: MockerFixture
):
    """Part 0 passed its discard check just before the user threw the take away."""
    upload, claim_row = await _late_part_zero_held_at_its_claim(user_id, mocker)

    await routes.discard_brain_dump(user_id=user_id, recording_id="rec-voice")
    claim_row.set()

    assert await _status_of(upload) == 409
    response = await service.finalize_typed_dump(
        user_id, "rec-typed", TYPED_TEXT, BackgroundTasks()
    )
    assert response.status == BrainDumpStatus.transcribed


async def test_a_late_first_part_claiming_the_row_mid_discard_gives_it_back(
    user_id: str, mocker: MockerFixture
):
    upload, claim_row = await _late_part_zero_held_at_its_claim(user_id, mocker)
    halfway, finish_discard = asyncio.Event(), asyncio.Event()
    held_mark, held_claim = _hold_the_first_to_return(
        routes.storage.mark_discarded,
        routes.db.claim_transition,
        reached=halfway,
        resume=finish_discard,
    )
    mocker.patch.object(routes.storage, "mark_discarded", new=held_mark)
    mocker.patch.object(routes.db, "claim_transition", new=held_claim)

    discard = asyncio.create_task(
        routes.discard_brain_dump(user_id=user_id, recording_id="rec-voice")
    )
    await asyncio.wait_for(halfway.wait(), timeout=5)
    claim_row.set()
    try:
        status = await _status_of(upload)
    finally:
        finish_discard.set()
        await discard

    assert status == 409
    response = await service.finalize_typed_dump(
        user_id, "rec-typed", TYPED_TEXT, BackgroundTasks()
    )
    assert response.status == BrainDumpStatus.transcribed


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


async def _late_part_zero_held_at_its_claim(
    user_id: str, mocker: MockerFixture
) -> tuple[asyncio.Task, asyncio.Event]:
    """Upload part 0 of ``rec-voice`` and hold it just before it claims the row."""
    checked, claim_row = asyncio.Event(), asyncio.Event()
    real_start_dump = db.start_dump

    async def held_start_dump(*args, **kwargs):
        checked.set()
        await claim_row.wait()
        return await real_start_dump(*args, **kwargs)

    mocker.patch.object(routes.db, "start_dump", new=held_start_dump)
    upload = asyncio.create_task(_upload_part_zero(user_id, "rec-voice"))
    await asyncio.wait_for(checked.wait(), timeout=5)
    return upload, claim_row


def _hold_the_first_to_return(*calls, reached: asyncio.Event, resume: asyncio.Event):
    """Wrap ``calls`` so whichever of them returns first waits for ``resume``."""
    returned = 0

    def wrap(real):
        async def held(*args, **kwargs):
            nonlocal returned
            result = await real(*args, **kwargs)
            returned += 1
            if returned == 1:
                reached.set()
                await resume.wait()
            return result

        return held

    return [wrap(call) for call in calls]


async def _status_of(upload: asyncio.Task) -> int:
    try:
        await upload
    except HTTPException as refused:
        return refused.status_code
    return 200
