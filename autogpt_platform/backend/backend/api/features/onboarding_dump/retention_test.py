"""The 48-hour sweep that deletes brain-dump voice recordings.

The bucket is a fake listing; ``clear_audio_paths`` is a recorder standing
in for the database manager, so the order of "clear the row" and "delete
the blob" is observable.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import retention, storage
from backend.util.gcs_utils_test import _URL_CONTAINING_404, _gcs_http_error

BUCKET = "private-bucket"
NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
# After every unmarked recording is gone: only the marker counts.
LATER = datetime(2027, 2, 1, 12, 0, tzinfo=timezone.utc)
MAX_AGE = timedelta(hours=42)
OLD = "uploads/users/user-1/uuid-a/brain-dump-rec-old.webm"
RECENT = "uploads/users/user-1/uuid-b/brain-dump-rec-new.webm"
MARKER = {storage.RECORDING_MARKER_KEY: storage.RECORDING_MARKER_VALUE}


def _item(
    name: str, age: timedelta, *, now: datetime = NOW, marked: bool = True
) -> dict[str, Any]:
    created = (now - age).isoformat().replace("+00:00", "Z")
    item: dict[str, Any] = {"name": name, "timeCreated": created}
    if marked:
        item["metadata"] = {"expires_at": "2026-01-01T00:00:00+00:00", **MARKER}
    return item


class Calls:
    def __init__(self) -> None:
        self.order: list[str] = []
        self.cleared: list[str] = []


@pytest.fixture
def calls() -> Calls:
    return Calls()


@pytest.fixture
def client(mocker: MockerFixture, calls: Calls) -> MagicMock:
    client = MagicMock()
    client.list_objects = AsyncMock(return_value={"items": []})

    async def delete(bucket: str, name: str) -> None:
        calls.order.append(f"delete:{name}")

    client.delete = AsyncMock(side_effect=delete)

    @asynccontextmanager
    async def fresh_client():
        yield client

    mocker.patch.object(retention, "fresh_gcs_client", fresh_client)
    mocker.patch.object(retention, "audio_max_age", return_value=MAX_AGE)
    handler = MagicMock()
    handler.config.gcs_bucket_name = BUCKET
    mocker.patch.object(
        retention, "get_cloud_storage_handler", AsyncMock(return_value=handler)
    )
    return client


def _clear(calls: Calls, rows: int = 1) -> AsyncMock:
    async def clear(paths: list[str]) -> int:
        calls.order.append("clear")
        calls.cleared.extend(paths)
        return rows

    return AsyncMock(side_effect=clear)


@pytest.mark.asyncio
async def test_old_recording_is_cleared_from_its_row_then_deleted(
    client: MagicMock, calls: Calls
):
    client.list_objects.return_value = {
        "items": [
            _item(OLD, MAX_AGE),
            _item(RECENT, MAX_AGE - timedelta(minutes=1)),
        ]
    }

    result = await retention.purge_expired_recordings(_clear(calls), now=NOW)

    assert result == retention.RecordingPurgeResult(
        expired=1, rows_cleared=1, deleted=1, failed=0
    )
    assert calls.cleared == [f"gcs://{BUCKET}/{OLD}"]
    assert calls.order == ["clear", f"delete:{OLD}"]
    params: dict[str, Any] = client.list_objects.call_args.kwargs["params"]
    assert params["prefix"] == "uploads/users/"


@pytest.mark.asyncio
async def test_recent_recordings_and_other_uploads_are_left_alone(
    client: MagicMock, calls: Calls
):
    """Old blobs without expiry metadata are found by age, but only
    recordings: a user's agent-input upload in the same prefix stays."""
    client.list_objects.return_value = {
        "items": [
            _item(RECENT, timedelta(days=1)),
            _item("uploads/users/user-1/uuid-c/report.pdf", timedelta(days=90)),
            _item("uploads/users/user-1/uuid-d/brain-dump-notes.txt", timedelta(90)),
            _item("uploads/users/user-1/brain-dump-x.webm", timedelta(days=90)),
        ]
    }
    clear = _clear(calls)

    result = await retention.purge_expired_recordings(clear, now=NOW)

    assert result == retention.RecordingPurgeResult()
    clear.assert_not_awaited()
    client.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_unmarked_recordings_from_before_the_marker_are_purged_by_name(
    client: MagicMock, calls: Calls
):
    """Recordings stored before ``store_audio`` marked them, orphans
    included, are only recognisable by their name."""
    client.list_objects.return_value = {
        "items": [_item(OLD, timedelta(days=30), marked=False)]
    }

    result = await retention.purge_expired_recordings(_clear(calls), now=NOW)

    assert result.deleted == 1
    assert calls.order == ["clear", f"delete:{OLD}"]


@pytest.mark.asyncio
async def test_a_user_upload_named_like_a_recording_is_left_alone(
    client: MagicMock, calls: Calls
):
    """A downloaded recording uploaded again through ``/files/upload`` keeps
    its name but not the marker, so schedules and presets using it keep
    working."""
    reuploaded = "uploads/users/user-1/uuid-g/brain-dump-rec-old.webm"
    client.list_objects.return_value = {
        "items": [
            _item(reuploaded, timedelta(days=20), now=LATER, marked=False),
            _item(OLD, timedelta(days=20), now=LATER),
        ]
    }
    clear = _clear(calls)

    result = await retention.purge_expired_recordings(clear, now=LATER)

    assert result.expired == 1
    assert calls.cleared == [f"gcs://{BUCKET}/{OLD}"]
    assert calls.order == ["clear", f"delete:{OLD}"]


def test_unmarked_recordings_are_recognised_until_2027():
    """Recordings are marked from this PR's release on. The cutoff has to be
    on or after that release, or an unmarked recording stored in between is
    never deleted."""
    assert retention._UNMARKED_RECORDINGS_BEFORE == datetime(
        2027, 1, 1, tzinfo=timezone.utc
    )


@pytest.mark.asyncio
async def test_an_unmarked_recording_created_at_the_cutoff_is_kept_and_counted(
    client: MagicMock, calls: Calls, mocker: MockerFixture
):
    """Kept, since an upload named like a recording looks the same; counted,
    so a release that landed after the cutoff shows up in the log."""
    caplog_info = mocker.patch.object(retention.logger, "info")
    cutoff = retention._UNMARKED_RECORDINGS_BEFORE
    now = cutoff + timedelta(days=10)
    at_cutoff = "uploads/users/user-3/uuid-h/brain-dump-rec-3.webm"
    client.list_objects.return_value = {
        "items": [_item(at_cutoff, now - cutoff, now=now, marked=False)]
    }
    clear = _clear(calls)

    result = await retention.purge_expired_recordings(clear, now=now)

    assert result == retention.RecordingPurgeResult(unmarked_skipped=1)
    clear.assert_not_awaited()
    client.delete.assert_not_awaited()
    logged = caplog_info.call_args.args[0]
    assert "unmarked_skipped=1" in logged
    assert "user-3" not in logged and "uploads/" not in logged


@pytest.mark.asyncio
async def test_a_marked_recording_newer_than_the_cutoff_is_still_taken(
    client: MagicMock, calls: Calls
):
    cutoff = retention._UNMARKED_RECORDINGS_BEFORE
    now = cutoff + timedelta(days=10)
    newer = "uploads/users/user-4/uuid-i/brain-dump-rec-4.webm"
    client.list_objects.return_value = {
        "items": [_item(newer, timedelta(days=5), now=now)]
    }

    result = await retention.purge_expired_recordings(_clear(calls), now=now)

    assert result == retention.RecordingPurgeResult(
        expired=1, rows_cleared=1, deleted=1, failed=0, unmarked_skipped=0
    )
    assert calls.order == ["clear", f"delete:{newer}"]


@pytest.mark.asyncio
async def test_a_large_backlog_is_spread_over_several_runs(
    client: MagicMock, calls: Calls, mocker: MockerFixture
):
    mocker.patch.object(retention, "_MAX_PURGED_PER_RUN", 1)
    client.list_objects.return_value = {
        "items": [_item(OLD, timedelta(days=60)), _item(RECENT, timedelta(days=61))]
    }

    result = await retention.purge_expired_recordings(_clear(calls), now=NOW)

    assert result.expired == 1
    assert calls.order == ["clear", f"delete:{OLD}"]


@pytest.mark.asyncio
async def test_a_newly_due_take_does_not_wait_behind_the_backlog(
    client: MagicMock, calls: Calls, mocker: MockerFixture
):
    """At first deploy every recording ever stored is due, in name (user ID)
    order. A failed take that just reached its max age must still go this
    run, or it outlives 48 hours waiting behind recordings long overdue."""
    mocker.patch.object(retention, "_MAX_PURGED_PER_RUN", 1)
    failed_take = "uploads/users/user-9/uuid-c/brain-dump-rec-failed.webm"
    client.list_objects.side_effect = [
        {"items": [_item(OLD, timedelta(days=60))], "nextPageToken": "page-2"},
        {"items": [_item(failed_take, MAX_AGE + timedelta(minutes=5))]},
    ]

    result = await retention.purge_expired_recordings(_clear(calls), now=NOW)

    assert result.expired == 1
    assert calls.order == ["clear", f"delete:{failed_take}"]


@pytest.mark.asyncio
async def test_blob_already_gone_counts_as_deleted(client: MagicMock, calls: Calls):
    client.list_objects.return_value = {"items": [_item(OLD, timedelta(days=30))]}
    client.delete.side_effect = _gcs_http_error(404, _URL_CONTAINING_404)

    result = await retention.purge_expired_recordings(_clear(calls), now=NOW)

    assert result.deleted == 1
    assert result.failed == 0


@pytest.mark.asyncio
async def test_failed_delete_is_counted_and_the_rest_still_go(
    client: MagicMock, calls: Calls
):
    other = "uploads/users/user-2/uuid-e/brain-dump-rec-2.m4a"
    client.list_objects.return_value = {
        "items": [_item(OLD, timedelta(days=8)), _item(other, timedelta(days=8))]
    }

    async def delete(bucket: str, name: str) -> None:
        if name == OLD:
            raise _gcs_http_error(503, _URL_CONTAINING_404)

    client.delete.side_effect = delete

    result = await retention.purge_expired_recordings(_clear(calls, rows=2), now=NOW)

    assert result == retention.RecordingPurgeResult(
        expired=2, rows_cleared=2, deleted=1, failed=1
    )


@pytest.mark.asyncio
async def test_nothing_is_deleted_when_the_rows_cannot_be_cleared(
    client: MagicMock,
):
    client.list_objects.return_value = {"items": [_item(OLD, timedelta(days=8))]}

    with pytest.raises(RuntimeError):
        await retention.purge_expired_recordings(
            AsyncMock(side_effect=RuntimeError("db down")), now=NOW
        )
    client.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_every_page_of_the_bucket_is_swept(client: MagicMock, calls: Calls):
    """The first run after this ships purges every recording ever stored."""
    second = "uploads/users/user-2/uuid-f/brain-dump-rec-2.webm"
    client.list_objects.side_effect = [
        {"items": [_item(OLD, timedelta(days=60))], "nextPageToken": "page-2"},
        {"items": [_item(second, timedelta(days=60))]},
    ]

    result = await retention.purge_expired_recordings(_clear(calls, rows=1), now=NOW)

    assert result.expired == 2
    assert result.deleted == 2
    assert client.list_objects.call_args.kwargs["params"]["pageToken"] == "page-2"


@pytest.mark.asyncio
async def test_no_bucket_means_nothing_to_purge(mocker: MockerFixture):
    handler = MagicMock()
    handler.config.gcs_bucket_name = ""
    mocker.patch.object(
        retention, "get_cloud_storage_handler", AsyncMock(return_value=handler)
    )
    clear = AsyncMock()

    assert await retention.purge_expired_recordings(clear) == (
        retention.RecordingPurgeResult()
    )
    clear.assert_not_awaited()
