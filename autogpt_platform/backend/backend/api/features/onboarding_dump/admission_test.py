from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks, HTTPException
from prisma.enums import BrainDumpInputMode, BrainDumpStatus

from backend.api.features.onboarding_dump import service
from backend.api.features.onboarding_dump.service_test import DumpStore


@pytest.fixture
def pipeline(mocker):
    store = DumpStore()
    for name, method in (
        ("get_dump", store.get_dump),
        ("start_dump", store.start_dump),
        ("claim_transition", store.claim_transition),
        ("mark_failed", store.mark_failed),
    ):
        mocker.patch.object(service.db, name, new=method)
    admission = mocker.patch.object(
        service, "enforce_personalization_budget", new=AsyncMock()
    )
    assemble = mocker.patch.object(
        service.storage, "assemble_parts", new=AsyncMock(return_value=b"")
    )
    discard = mocker.patch.object(service.storage, "discard_parts", new=AsyncMock())
    quality = mocker.patch.object(
        service.quality, "check_transcript_quality", new=AsyncMock(return_value=None)
    )
    return store, admission, assemble, discard, quality


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("mode", [BrainDumpInputMode.voice, BrainDumpInputMode.typed])
async def test_rejected_processing_keeps_recording_and_can_retry(
    pipeline, status, mode
):
    store, admission, assemble, discard, quality = pipeline
    await store.start_dump("user", "recording", mode)
    admission.side_effect = HTTPException(
        status, "Try again", headers={"Retry-After": "30"}
    )
    tasks = BackgroundTasks()

    async def finalize():
        if mode == BrainDumpInputMode.voice:
            return await service.finalize_voice_dump(
                "user", "recording", 12, None, tasks
            )
        return await service.finalize_typed_dump(
            "user", "recording", "My bakery", tasks
        )

    with pytest.raises(HTTPException) as caught:
        await finalize()
    assert caught.value.status_code == status
    assert store.row is not None
    assert store.row.status == BrainDumpStatus.failed
    assert store.row.recordingId == "recording"
    assemble.assert_not_awaited()
    discard.assert_not_awaited()
    quality.assert_not_awaited()
    assert tasks.tasks == []

    admission.side_effect = None
    if mode == BrainDumpInputMode.voice:
        # The voice retry replays part zero before finalizing the same take.
        await store.start_dump("user", "recording", mode)
    await finalize()
    assert admission.await_count == 2
    if mode == BrainDumpInputMode.voice:
        assemble.assert_awaited_once()
    else:
        assert len(tasks.tasks) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [BrainDumpStatus.transcribing, BrainDumpStatus.completed]
)
@pytest.mark.parametrize("mode", [BrainDumpInputMode.voice, BrainDumpInputMode.typed])
async def test_existing_processing_and_results_bypass_budget(pipeline, status, mode):
    store, admission, _, _, _ = pipeline
    row = await store.start_dump("user", "recording", mode)
    row.status = status
    if mode == BrainDumpInputMode.voice:
        await service.finalize_voice_dump(
            "user", "recording", 12, None, BackgroundTasks()
        )
    else:
        await service.finalize_typed_dump(
            "user", "recording", "My bakery", BackgroundTasks()
        )
    admission.assert_not_awaited()


@pytest.mark.asyncio
async def test_skip_bypasses_budget(pipeline, mocker):
    store, admission, _, _, _ = pipeline
    mocker.patch.object(service.db, "update_dump", new=store.update_dump)
    await service.finalize_skipped_dump("user", "recording")
    admission.assert_not_awaited()
