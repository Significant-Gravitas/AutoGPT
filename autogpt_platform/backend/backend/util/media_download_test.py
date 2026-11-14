"""
Tests for media URL downloading (media_download.py).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.file import store_media_file
from backend.util.media_download import download_media_url
from backend.util.request import HTTPServerError, Requests, Response
from backend.util.type import MediaFileType


def _mock_response(status: int, reason: str | None, url: str) -> Response:
    """Build a ``Response`` suitable for mocking ``Requests._request``."""
    client_response = MagicMock(
        status=status,
        reason=reason,
        headers={"content-type": "text/html"},
        request_info=None,
    )
    return Response(
        response=client_response,
        url=url,
        body=b"<!DOCTYPE html>",
    )


def make_test_context(
    graph_exec_id: str = "test-exec-123",
    user_id: str = "test-user-123",
):
    from backend.data.execution import ExecutionContext

    return ExecutionContext(
        user_id=user_id,
        graph_exec_id=graph_exec_id,
    )


# ---------------------------------------------------------------------------
# download_media_url unit tests
# ---------------------------------------------------------------------------


class TestDownloadMediaUrl:
    @pytest.mark.asyncio
    async def test_http_400_raises_value_error(self):
        """HTTP 400 (bad request / rejected URL) is raised as ValueError."""
        url = "https://upload.wikimedia.org/thumb/a/ab/Cat.jpg/333px-Cat.jpg"
        reason = "Use thumbnail sizes listed on https://w.wiki/GHai"
        response = _mock_response(400, reason, url)

        with patch.object(
            Requests, "_request", AsyncMock(return_value=response)
        ) as mock_req:
            with pytest.raises(ValueError) as exc_info:
                await download_media_url(url)

        assert mock_req.call_count == 1
        assert str(exc_info.value) == (
            f"Could not download file from {url}: HTTP 400 ({reason})"
        )

    @pytest.mark.asyncio
    async def test_http_400_without_reason(self):
        """HTTP 400 with no reason phrase still raises a clean ValueError."""
        url = "https://example.com/missing.png"
        response = _mock_response(400, None, url)

        with patch.object(Requests, "_request", AsyncMock(return_value=response)):
            with pytest.raises(ValueError) as exc_info:
                await download_media_url(url)

        assert str(exc_info.value) == (f"Could not download file from {url}: HTTP 400")

    @pytest.mark.asyncio
    async def test_http_401_raises_http_client_error(self):
        """HTTP 401 is NOT converted to ValueError; it propagates as HTTPClientError
        so that bad/expired credentials reach Sentry."""
        url = "https://example.com/secret.mp4"
        response = _mock_response(401, "Unauthorized", url)

        with patch.object(Requests, "_request", AsyncMock(return_value=response)):
            from backend.util.request import HTTPClientError

            with pytest.raises(HTTPClientError):
                await download_media_url(url)

    @pytest.mark.asyncio
    async def test_http_403_raises_http_client_error(self):
        """HTTP 403 is NOT converted to ValueError; it propagates as HTTPClientError."""
        url = "https://example.com/forbidden.mp4"
        response = _mock_response(403, "Forbidden", url)

        with patch.object(Requests, "_request", AsyncMock(return_value=response)):
            from backend.util.request import HTTPClientError

            with pytest.raises(HTTPClientError):
                await download_media_url(url)

    @pytest.mark.asyncio
    async def test_http_503_raises_http_server_error(self):
        """HTTP 503 propagates as HTTPServerError (not ValueError)."""
        url = "https://example.com/image.png"
        response = _mock_response(503, "Service Unavailable", url)

        with patch.object(
            Requests, "_request", AsyncMock(return_value=response)
        ) as mock_req:
            with pytest.raises(HTTPServerError):
                await download_media_url(url)

        assert mock_req.call_count == 1


# ---------------------------------------------------------------------------
# store_media_file URL integration
# ---------------------------------------------------------------------------


class TestStoreMediaFileUrlErrors:
    @pytest.fixture(autouse=True)
    def _no_cloud_storage(self):
        handler = MagicMock()
        handler.is_cloud_path.return_value = False
        with patch("backend.util.file.get_cloud_storage_handler", return_value=handler):
            yield

    @pytest.mark.asyncio
    async def test_client_error_is_reported_as_bad_input(self):
        """A URL the server rejects (#15340: Wikimedia thumbnail at a
        non-standard width) is the caller's to fix, so it must be a ValueError
        that blocks report as a handled error, not an unknown platform error."""
        url = "https://upload.wikimedia.org/thumb/a/ab/Cat.jpg/333px-Cat.jpg"
        reason = "Use thumbnail sizes listed on https://w.wiki/GHai"
        response = _mock_response(400, reason, url)

        with patch.object(
            Requests, "_request", AsyncMock(return_value=response)
        ) as mock_req:
            with pytest.raises(ValueError) as exc_info:
                await store_media_file(
                    file=MediaFileType(url),
                    execution_context=make_test_context(),
                    return_format="for_local_processing",
                )

        assert mock_req.call_count == 1
        from backend.util.request import HTTPClientError

        assert not isinstance(exc_info.value, HTTPClientError)
        assert str(exc_info.value) == (
            f"Could not download file from {url}: HTTP 400 ({reason})"
        )

    @pytest.mark.asyncio
    async def test_throttled_client_error_fails_immediately_without_retrying(self):
        """HTTP 429 is in Requests' default THROTTLE_RETRY_STATUS_CODES but is
        not HTTP 400, so it propagates as HTTPClientError (not ValueError) and
        still fails immediately because retry_max_attempts=1."""
        url = "https://example.com/throttled.png"
        reason = "Too Many Requests"
        response = _mock_response(429, reason, url)

        with patch.object(
            Requests, "_request", AsyncMock(return_value=response)
        ) as mock_req:
            from backend.util.request import HTTPClientError

            with pytest.raises(HTTPClientError):
                await store_media_file(
                    file=MediaFileType(url),
                    execution_context=make_test_context(),
                    return_format="for_local_processing",
                )

        assert mock_req.call_count == 1

    @pytest.mark.asyncio
    async def test_server_error_keeps_its_exception_and_does_not_retry_forever(self):
        """HTTP 503 is in THROTTLE_RETRY_STATUS_CODES. The real Requests retry
        wrapper must be bounded by retry_max_attempts=1 and re-raise
        HTTPServerError on attempt 1 without hanging indefinitely."""
        url = "https://example.com/image.png"
        response = _mock_response(503, "Service Unavailable", url)

        with patch.object(
            Requests, "_request", AsyncMock(return_value=response)
        ) as mock_req:
            with pytest.raises(HTTPServerError):
                await store_media_file(
                    file=MediaFileType(url),
                    execution_context=make_test_context(),
                    return_format="for_local_processing",
                )

        assert mock_req.call_count == 1
