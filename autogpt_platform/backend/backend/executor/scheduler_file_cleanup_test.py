"""The scheduled cloud-storage cleanup: voice recordings, then the rest."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump.retention import RecordingPurgeResult
from backend.executor import scheduler


def _run(coro):
    return asyncio.run(coro)


def test_recordings_are_purged_before_the_generic_cleanup(mocker: MockerFixture):
    order: list[str] = []
    db = MagicMock()
    mocker.patch.object(scheduler, "get_database_manager_async_client", return_value=db)

    async def purge(clear):
        order.append("recordings")
        assert clear is db.clear_brain_dump_audio_paths
        return RecordingPurgeResult()

    async def generic():
        order.append("generic")
        return 0

    mocker.patch.object(scheduler, "purge_expired_recordings", side_effect=purge)
    mocker.patch.object(scheduler, "cleanup_expired_files_async", side_effect=generic)
    mocker.patch.object(scheduler, "run_async", side_effect=_run)

    scheduler.cleanup_expired_files()

    assert order == ["recordings", "generic"]


def test_a_failed_recording_purge_does_not_stop_the_generic_cleanup(
    mocker: MockerFixture,
):
    mocker.patch.object(scheduler, "get_database_manager_async_client")
    mocker.patch.object(
        scheduler,
        "purge_expired_recordings",
        AsyncMock(side_effect=RuntimeError("gcs down")),
    )
    generic = mocker.patch.object(
        scheduler, "cleanup_expired_files_async", AsyncMock(return_value=0)
    )
    mocker.patch.object(scheduler, "run_async", side_effect=_run)

    scheduler.cleanup_expired_files()

    generic.assert_awaited_once()
