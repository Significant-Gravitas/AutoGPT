import asyncio
import functools
import random
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from google.auth.credentials import Signing
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from backend.util.gcs_utils import (
    blob_created_at,
    download_in_chunks,
    download_range,
    download_with_fresh_session,
    generate_iam_signed_url,
    parse_gcs_path,
)


def _mock_client(mocker, *, download: AsyncMock) -> MagicMock:
    """Patch the async GCS Storage client and session used by gcs_utils."""
    client = MagicMock()
    client.download = download
    client.close = AsyncMock(return_value=None)
    mocker.patch(
        "backend.util.gcs_utils.async_gcs_storage.Storage",
        return_value=client,
    )
    session = MagicMock()
    session.close = AsyncMock(return_value=None)
    mocker.patch(
        "backend.util.gcs_utils.aiohttp.ClientSession",
        return_value=session,
    )
    mocker.patch(
        "backend.util.gcs_utils.aiohttp.TCPConnector", return_value=MagicMock()
    )
    return client


def test_parse_gcs_path_splits_bucket_and_blob():
    assert parse_gcs_path("gcs://my-bucket/path/to/file") == (
        "my-bucket",
        "path/to/file",
    )


def test_parse_gcs_path_rejects_invalid_prefix():
    with pytest.raises(ValueError):
        parse_gcs_path("s3://my-bucket/file")


@pytest.mark.asyncio
async def test_generate_iam_signed_url_uses_adc_service_account(mocker):
    credentials = MagicMock()
    credentials.valid = False
    credentials.service_account_email = "default"

    def refresh(_request):
        credentials.valid = True
        credentials.service_account_email = "backend@example.iam.gserviceaccount.com"
        credentials.token = "access-token"

    credentials.refresh.side_effect = refresh
    mocker.patch(
        "backend.util.gcs_utils.google_auth_default",
        return_value=(credentials, "project"),
    )
    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.generate_signed_url.return_value = "https://signed.example/object"

    result = await generate_iam_signed_url(client, "bucket", "object", 60)

    assert result == "https://signed.example/object"
    credentials.refresh.assert_called_once()
    assert blob.generate_signed_url.call_args.kwargs["service_account_email"] == (
        "backend@example.iam.gserviceaccount.com"
    )
    assert blob.generate_signed_url.call_args.kwargs["access_token"] == "access-token"
    assert blob.generate_signed_url.call_args.kwargs["version"] == "v4"
    assert blob.generate_signed_url.call_args.kwargs["method"] == "GET"


@pytest.mark.asyncio
async def test_generate_iam_signed_url_uses_locally_signing_credentials(mocker):
    credentials = MagicMock(spec=Signing)
    mocker.patch(
        "backend.util.gcs_utils.google_auth_default",
        return_value=(credentials, "project"),
    )
    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.generate_signed_url.return_value = "https://signed.example/object"

    result = await generate_iam_signed_url(client, "bucket", "object", 60)

    assert result == "https://signed.example/object"
    blob.generate_signed_url.assert_called_once()
    assert blob.generate_signed_url.call_args.kwargs["credentials"] is credentials
    assert "service_account_email" not in blob.generate_signed_url.call_args.kwargs
    assert "access_token" not in blob.generate_signed_url.call_args.kwargs


@pytest.mark.asyncio
async def test_download_range_sends_range_header_and_slices(mocker):
    download = AsyncMock(return_value=b"0123456789ABCDEF")
    client = _mock_client(mocker, download=download)

    result = await download_range("bucket", "blob", 8)

    assert result == b"01234567"
    download.assert_awaited_once_with("bucket", "blob", headers={"Range": "bytes=0-7"})
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_download_range_falls_back_when_headers_unsupported(mocker):
    download = AsyncMock(
        side_effect=[TypeError("unexpected kwarg 'headers'"), b"full-content-here"]
    )
    _mock_client(mocker, download=download)

    result = await download_range("bucket", "blob", 4)

    assert result == b"full"
    assert download.await_count == 2


def _gcs_http_error(status: int, url: str) -> aiohttp.ClientResponseError:
    """The error gcloud-aio raises for a non-2xx GCS response."""
    request_info = aiohttp.RequestInfo(
        url=URL(url), method="GET", headers=CIMultiDictProxy(CIMultiDict())
    )
    return aiohttp.ClientResponseError(
        request_info, (), status=status, message="GCS error"
    )


# File ids are UUIDs, so "404" turns up in object URLs by chance.
_URL_CONTAINING_404 = (
    "https://storage.googleapis.com/storage/v1/b/bucket/o/"
    "workspaces%2Fws%2F1aa50e64-f289-404d-889a-935d443c0ca0%2Fa.png?alt=media"
)


@pytest.mark.asyncio
async def test_download_range_maps_404_to_file_not_found(mocker):
    download = AsyncMock(side_effect=_gcs_http_error(404, _URL_CONTAINING_404))
    _mock_client(mocker, download=download)

    with pytest.raises(FileNotFoundError):
        await download_range("bucket", "missing", 16)


@pytest.mark.asyncio
async def test_download_maps_404_to_file_not_found(mocker):
    download = AsyncMock(side_effect=_gcs_http_error(404, _URL_CONTAINING_404))
    _mock_client(mocker, download=download)

    with pytest.raises(FileNotFoundError):
        await download_with_fresh_session("bucket", "missing")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 500, 503])
async def test_download_keeps_other_errors_when_url_contains_404(mocker, status):
    download = AsyncMock(side_effect=_gcs_http_error(status, _URL_CONTAINING_404))
    _mock_client(mocker, download=download)

    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        await download_with_fresh_session("bucket", "blob")
    assert exc_info.value.status == status

    with pytest.raises(aiohttp.ClientResponseError):
        await download_range("bucket", "blob", 16)


@pytest.mark.asyncio
async def test_download_in_chunks_reads_the_object_as_it_is_consumed(mocker):
    body = StreamResponseStub(b"abcdefgh")
    client = _mock_client(mocker, download=AsyncMock(side_effect=AssertionError))
    client.download_stream = AsyncMock(return_value=body)

    chunks = download_in_chunks("bucket", "blob", 3)
    first = await anext(chunks)

    assert first == b"abc"
    assert body.read_sizes == [3]
    assert [first] + [chunk async for chunk in chunks] == [b"abc", b"def", b"gh"]
    # timeout=0 is no total timeout in aiohttp, so a long download is not cut off.
    assert client.download_stream.await_args.kwargs["timeout"] == 0


@pytest.mark.asyncio
async def test_download_in_chunks_maps_404_to_file_not_found(mocker):
    client = _mock_client(mocker, download=AsyncMock())
    client.download_stream = AsyncMock(
        side_effect=_gcs_http_error(404, _URL_CONTAINING_404)
    )

    with pytest.raises(FileNotFoundError):
        await anext(download_in_chunks("bucket", "missing", 16))


@pytest.mark.asyncio
async def test_download_in_chunks_resumes_where_the_connection_dropped(dropping_gcs):
    gcs = await dropping_gcs(drops=1)

    chunks = download_in_chunks("bucket", "blob", 1024)
    first = await anext(chunks)
    gcs.drop.set()
    content = first + b"".join([chunk async for chunk in chunks])

    assert content == _CONTENT
    assert gcs.ranges[0] is None
    assert len(gcs.ranges) == 2 and gcs.ranges[1].startswith("bytes=")


@pytest.mark.asyncio
async def test_download_in_chunks_stops_resuming_a_connection_that_keeps_dropping(
    dropping_gcs,
):
    gcs = await dropping_gcs(drops=4)

    chunks = download_in_chunks("bucket", "blob", 1024)
    await anext(chunks)
    gcs.drop.set()
    with pytest.raises(aiohttp.ClientPayloadError):
        async for _ in chunks:
            pass

    assert len(gcs.ranges) == 3


@pytest.mark.asyncio
async def test_download_in_chunks_refuses_a_resume_that_ignores_the_range(
    dropping_gcs,
):
    gcs = await dropping_gcs(drops=1, honour_range=False)

    received = []
    chunks = download_in_chunks("bucket", "blob", 1024)
    received.append(await anext(chunks))
    gcs.drop.set()
    with pytest.raises(RuntimeError, match="resumed"):
        async for chunk in chunks:
            received.append(chunk)

    assert _CONTENT.startswith(b"".join(received))


@pytest.mark.asyncio
async def test_download_in_chunks_retries_a_refused_resume_without_logging_credentials(
    dropping_gcs, monkeypatch, caplog
):
    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        functools.partial(
            aiohttp.ClientSession, headers={"Authorization": "Bearer test-token"}
        ),
    )
    gcs = await dropping_gcs(drops=1, refuse=1)

    chunks = download_in_chunks("bucket", "blob", 1024)
    with caplog.at_level("WARNING"):
        first = await anext(chunks)
        gcs.drop.set()
        content = first + b"".join([chunk async for chunk in chunks])

    assert content == _CONTENT
    assert len(gcs.ranges) == 3
    assert gcs.authorizations[1] == "bearer test-token"
    assert "ClientResponseError" in caplog.text
    assert "test-token" not in caplog.text


class StreamResponseStub:
    def __init__(self, content: bytes):
        self._content = content
        self.read_sizes: list[int] = []
        self.content_length = len(content)

    async def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        chunk, self._content = self._content[:size], self._content[size:]
        return chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info) -> None:
        pass


# Not periodic, so bytes served from the wrong offset never match by chance.
_CONTENT = random.Random(0).randbytes(204_800)
_CUT = 50_000


@pytest.fixture
async def dropping_gcs(monkeypatch):
    """Start a ``DroppingGCS`` and point the GCS client at it."""
    started: list[tuple[DroppingGCS, asyncio.Server]] = []

    async def start(**kwargs) -> DroppingGCS:
        gcs = DroppingGCS(**kwargs)
        server = await asyncio.start_server(gcs.serve, "127.0.0.1", 0)
        started.append((gcs, server))
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("STORAGE_EMULATOR_HOST", f"http://127.0.0.1:{port}")
        return gcs

    yield start
    for gcs, server in started:
        gcs.drop.set()
        server.close()
        await server.wait_closed()


class DroppingGCS:
    """GCS's media endpoint over real TCP. It honours ``Range: bytes=N-`` and
    closes the connection ``_CUT`` bytes into each of its first ``drops``
    bodies, once ``drop`` is set, as GCS did mid-download in production.
    The first ``refuse`` requests after the first answer 503."""

    def __init__(self, drops: int, honour_range: bool = True, refuse: int = 0):
        self.drops = drops
        self.honour_range = honour_range
        self.refuse = refuse
        self.ranges: list[str | None] = []
        self.authorizations: list[str | None] = []
        self.drop = asyncio.Event()

    async def serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        head = (await reader.readuntil(b"\r\n\r\n")).decode()
        headers = dict(
            line.lower().split(": ", 1) for line in head.split("\r\n")[1:] if line
        )
        range_header = headers.get("range")
        self.ranges.append(range_header)
        self.authorizations.append(headers.get("authorization"))
        if 1 < len(self.ranges) <= 1 + self.refuse:
            writer.write(
                b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n"
            )
            await writer.drain()
            writer.close()
            return
        start = 0
        if range_header and self.honour_range:
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
        body = _CONTENT[start:]
        status = "206 Partial Content" if start else "200 OK"
        writer.write(
            f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        )
        if len(self.ranges) <= self.drops:
            writer.write(body[:_CUT])
            await writer.drain()
            await self.drop.wait()
        else:
            writer.write(body)
        await writer.drain()
        writer.close()


@pytest.mark.asyncio
async def test_blob_created_at_reads_metadata_only(mocker):
    client = _mock_client(mocker, download=AsyncMock())
    client.download_metadata = AsyncMock(
        return_value={"name": "blob", "timeCreated": "2026-10-13T12:00:00.123Z"}
    )

    assert await blob_created_at("bucket", "blob") == datetime(
        2026, 10, 13, 12, 0, 0, 123000, tzinfo=timezone.utc
    )
    client.download_metadata.assert_awaited_once_with("bucket", "blob")
    client.download.assert_not_awaited()
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_blob_created_at_is_none_only_for_a_real_404(mocker):
    client = _mock_client(mocker, download=AsyncMock())
    client.download_metadata = AsyncMock(
        side_effect=_gcs_http_error(404, _URL_CONTAINING_404)
    )
    assert await blob_created_at("bucket", "missing") is None

    client.download_metadata = AsyncMock(
        side_effect=_gcs_http_error(503, _URL_CONTAINING_404)
    )
    with pytest.raises(aiohttp.ClientResponseError):
        await blob_created_at("bucket", "blob")
