import asyncio
import contextlib
import re
import struct
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import aioclamd
import pytest
from pydantic import ValidationError

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.util import virus_scanner
from backend.util.settings import Config
from backend.util.virus_scanner import (
    VirusScannerService,
    VirusScannerSettings,
    VirusScanResult,
    get_virus_scanner,
    scan_content_safe,
)


class TestVirusScannerService:
    @pytest.fixture
    def scanner_settings(self):
        return VirusScannerSettings(
            clamav_service_host="localhost",
            clamav_service_port=3310,
            clamav_service_enabled=True,
            max_scan_size=10 * 1024 * 1024,  # 10MB for testing
            mark_failed_scans_as_clean=False,  # For testing, failed scans should be clean
        )

    @pytest.fixture
    def scanner(self, scanner_settings):
        return VirusScannerService(scanner_settings)

    @pytest.fixture
    def disabled_scanner(self):
        settings = VirusScannerSettings(clamav_service_enabled=False)
        return VirusScannerService(settings)

    def test_scanner_initialization(self, scanner_settings):
        scanner = VirusScannerService(scanner_settings)
        assert scanner.settings.clamav_service_host == "localhost"
        assert scanner.settings.clamav_service_port == 3310
        assert scanner.settings.clamav_service_enabled is True

    @pytest.mark.asyncio
    async def test_scan_disabled_returns_clean(self, disabled_scanner):
        content = b"test file content"
        result = await disabled_scanner.scan_file(content, filename="test.txt")

        assert result.is_clean is True
        assert result.threat_name is None
        assert result.file_size == len(content)
        assert result.scan_time_ms == 0

    @pytest.mark.asyncio
    async def test_scan_file_too_large(self, scanner):
        # Create content larger than max_scan_size
        large_content = b"x" * (scanner.settings.max_scan_size + 1)

        # Large files behavior depends on mark_failed_scans_as_clean setting
        result = await scanner.scan_file(large_content, filename="large_file.txt")
        assert result.is_clean == scanner.settings.mark_failed_scans_as_clean
        assert result.file_size == len(large_content)
        assert result.scan_time_ms == 0

    @pytest.mark.asyncio
    async def test_scan_file_too_large_both_configurations(self):
        """Test large file handling with both mark_failed_scans_as_clean configurations"""
        large_content = b"x" * (10 * 1024 * 1024 + 1)  # Larger than 10MB

        # Test with mark_failed_scans_as_clean=True
        settings_clean = VirusScannerSettings(
            max_scan_size=10 * 1024 * 1024, mark_failed_scans_as_clean=True
        )
        scanner_clean = VirusScannerService(settings_clean)
        result_clean = await scanner_clean.scan_file(
            large_content, filename="large_file.txt"
        )
        assert result_clean.is_clean is True

        # Test with mark_failed_scans_as_clean=False
        settings_dirty = VirusScannerSettings(
            max_scan_size=10 * 1024 * 1024, mark_failed_scans_as_clean=False
        )
        scanner_dirty = VirusScannerService(settings_dirty)
        result_dirty = await scanner_dirty.scan_file(
            large_content, filename="large_file.txt"
        )
        assert result_dirty.is_clean is False

    @pytest.mark.asyncio
    async def test_scan_empty_file(self, scanner):
        """Empty files (0 bytes) should be accepted without hitting ClamAV."""
        content = b""
        result = await scanner.scan_file(content, filename="empty.png")

        assert result.is_clean is True
        assert result.threat_name is None
        assert result.file_size == 0
        assert result.scan_time_ms == 0

    @pytest.mark.asyncio
    async def test_scan_single_byte_file(self, scanner):
        """A 1-byte file should be scanned normally (regression: chunk_size must not be 0)."""

        async def mock_instream(*_):
            await asyncio.sleep(0.001)
            return {"stream": ("OK", None)}

        mock_client = Mock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.instream = AsyncMock(side_effect=mock_instream)
        scanner._client = mock_client

        content = b"\x00"
        result = await scanner.scan_file(content, filename="tiny.bin")

        assert result.is_clean is True
        assert result.file_size == 1
        assert result.scan_time_ms > 0

    @pytest.mark.asyncio
    async def test_scan_clean_file(self, scanner):
        async def mock_instream(*_):
            await asyncio.sleep(0.001)  # Small delay to ensure timing > 0
            return {"stream": ("OK", None)}

        mock_client = Mock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.instream = AsyncMock(side_effect=mock_instream)

        # Replace the client instance that was created in the constructor
        scanner._client = mock_client

        content = b"clean file content"
        result = await scanner.scan_file(content, filename="clean.txt")

        assert result.is_clean is True
        assert result.threat_name is None
        assert result.file_size == len(content)
        assert result.scan_time_ms > 0

    @pytest.mark.asyncio
    async def test_a_client_that_returns_no_result_fails_the_scan(self, scanner):
        mock_client = Mock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.instream = AsyncMock(return_value=None)
        scanner._client = mock_client

        with pytest.raises(aioclamd.ResponseError, match="no result"):
            await scanner.scan_file(b"some upload", filename="upload.bin")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "closed", [BrokenPipeError, ConnectionAbortedError, ConnectionResetError]
    )
    async def test_a_connection_closed_mid_scan_fails_the_scan(self, scanner, closed):
        mock_client = Mock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.instream = AsyncMock(side_effect=closed())
        scanner._client = mock_client

        with pytest.raises(RuntimeError, match="upload.bin .* was not scanned"):
            await scanner.scan_file(b"some upload", filename="upload.bin")

    @pytest.mark.asyncio
    async def test_scan_infected_file(self, scanner):
        async def mock_instream(*_):
            await asyncio.sleep(0.001)  # Small delay to ensure timing > 0
            return {"stream": ("FOUND", "Win.Test.EICAR_HDB-1")}

        mock_client = Mock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.instream = AsyncMock(side_effect=mock_instream)

        # Replace the client instance that was created in the constructor
        scanner._client = mock_client

        content = b"infected file content"
        result = await scanner.scan_file(content, filename="infected.txt")

        assert result.is_clean is False
        assert result.threat_name == "Win.Test.EICAR_HDB-1"
        assert result.file_size == len(content)
        assert result.scan_time_ms > 0

    @pytest.mark.asyncio
    async def test_scan_clamav_unavailable_fail_safe(self, scanner):
        mock_client = Mock()
        mock_client.ping = AsyncMock(return_value=False)

        # Replace the client instance that was created in the constructor
        scanner._client = mock_client

        content = b"test content"

        with pytest.raises(RuntimeError, match="ClamAV service is unreachable"):
            await scanner.scan_file(content, filename="test.txt")

    @pytest.mark.asyncio
    async def test_scan_error_fail_safe(self, scanner):
        mock_client = Mock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.instream = AsyncMock(side_effect=Exception("Scanning error"))

        # Replace the client instance that was created in the constructor
        scanner._client = mock_client

        content = b"test content"

        with pytest.raises(Exception, match="Scanning error"):
            await scanner.scan_file(content, filename="test.txt")

    # Note: scan_file_method and scan_upload_file tests removed as these APIs don't exist in current implementation

    def test_get_virus_scanner_singleton(self):
        scanner1 = get_virus_scanner()
        scanner2 = get_virus_scanner()

        # Should return the same instance
        assert scanner1 is scanner2

    # Note: client_reuse test removed as _get_client method doesn't exist in current implementation

    def test_scan_result_model(self):
        # Test VirusScanResult model
        result = VirusScanResult(
            is_clean=False, threat_name="Test.Virus", scan_time_ms=150, file_size=1024
        )

        assert result.is_clean is False
        assert result.threat_name == "Test.Virus"
        assert result.scan_time_ms == 150
        assert result.file_size == 1024

    @pytest.mark.asyncio
    async def test_concurrent_scans(self, scanner):
        async def mock_instream(*_):
            await asyncio.sleep(0.001)  # Small delay to ensure timing > 0
            return {"stream": ("OK", None)}

        mock_client = Mock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.instream = AsyncMock(side_effect=mock_instream)

        # Replace the client instance that was created in the constructor
        scanner._client = mock_client

        content1 = b"file1 content"
        content2 = b"file2 content"

        # Run concurrent scans
        results = await asyncio.gather(
            scanner.scan_file(content1, filename="file1.txt"),
            scanner.scan_file(content2, filename="file2.txt"),
        )

        assert len(results) == 2
        assert all(result.is_clean for result in results)
        assert results[0].file_size == len(content1)
        assert results[1].file_size == len(content2)
        assert all(result.scan_time_ms > 0 for result in results)


class TestHelperFunctions:
    """Test the helper functions scan_content_safe"""

    @pytest.mark.asyncio
    async def test_scan_content_safe_clean(self):
        """Test scan_content_safe with clean content"""
        with patch("backend.util.virus_scanner.get_virus_scanner") as mock_get_scanner:
            mock_scanner = Mock()
            mock_scanner.scan_file = AsyncMock()
            mock_scanner.scan_file.return_value = Mock(
                is_clean=True, threat_name=None, scan_time_ms=50, file_size=100
            )
            mock_get_scanner.return_value = mock_scanner

            # Should not raise any exception
            await scan_content_safe(b"clean content", filename="test.txt")

    @pytest.mark.asyncio
    async def test_scan_content_safe_infected(self):
        """Test scan_content_safe with infected content"""
        with patch("backend.util.virus_scanner.get_virus_scanner") as mock_get_scanner:
            mock_scanner = Mock()
            mock_scanner.scan_file = AsyncMock()
            mock_scanner.scan_file.return_value = Mock(
                is_clean=False, threat_name="Test.Virus", scan_time_ms=50, file_size=100
            )
            mock_get_scanner.return_value = mock_scanner

            with pytest.raises(VirusDetectedError) as exc_info:
                await scan_content_safe(b"infected content", filename="virus.txt")

            assert exc_info.value.threat_name == "Test.Virus"

    @pytest.mark.asyncio
    async def test_scan_content_safe_scan_error(self):
        """Test scan_content_safe when scanning fails"""
        with patch("backend.util.virus_scanner.get_virus_scanner") as mock_get_scanner:
            mock_scanner = Mock()
            mock_scanner.scan_file = AsyncMock()
            mock_scanner.scan_file.side_effect = Exception("Scan failed")
            mock_get_scanner.return_value = mock_scanner

            with pytest.raises(VirusScanError, match="Virus scanning failed"):
                await scan_content_safe(b"test content", filename="test.txt")

    @pytest.mark.asyncio
    async def test_scan_content_safe_logs_warning_not_error_on_failure(self):
        """Scan failures should log at WARNING level, not ERROR, to avoid paging on-call."""
        with patch("backend.util.virus_scanner.get_virus_scanner") as mock_get_scanner:
            mock_scanner = Mock()
            mock_scanner.scan_file = AsyncMock()
            mock_scanner.scan_file.side_effect = Exception(
                "range() arg 3 must not be zero"
            )
            mock_get_scanner.return_value = mock_scanner

            with (
                pytest.raises(VirusScanError),
                patch("backend.util.virus_scanner.logger") as mock_logger,
            ):
                await scan_content_safe(b"test", filename="screenshot.png")

            mock_logger.warning.assert_called_once()
            # Check the formatted log message contains the error text.
            # Use str() to handle both f-string and %-style logging formats.
            log_msg = str(mock_logger.warning.call_args)
            assert "range()" in log_msg
            mock_logger.error.assert_not_called()


class TestStreamingToClamd:
    """Drives the real client against a ClamAV stand-in on a local socket, served
    by the test's own event loop, so a scan that never yields starves it."""

    @pytest.mark.asyncio
    async def test_streaming_a_large_file_leaves_the_event_loop_free(self):
        async with _fake_clamd() as port:
            scanner = VirusScannerService(
                VirusScannerSettings(clamav_service_port=port)
            )
            result, longest_stall = await _with_longest_stall(
                scanner.scan_file(b"\x00" * (24 * 1024 * 1024), filename="clip.mp4")
            )

        assert result.is_clean
        assert longest_stall < 2.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("read_past_limit", "error"),
        [
            # A refusal the client reads names the size.
            (True, "too large to scan"),
            # A real clamd closes the stream it refused, so the client may see
            # the close before the reply.
            (False, "too large to scan|was not scanned"),
        ],
        ids=["clamd-reads-on", "clamd-closes"],
    )
    async def test_a_file_over_clamds_stream_limit_is_refused_not_scanned_in_pieces(
        self, monkeypatch, read_past_limit, error
    ):
        sessions: list[int] = []
        async with _fake_clamd(
            stream_max=_MiB,
            read_past_limit=read_past_limit,
            found_marker=_MARKER,
            sessions=sessions,
        ) as port:
            scanner = VirusScannerService(
                VirusScannerSettings(
                    clamav_service_port=port, mark_failed_scans_as_clean=True
                )
            )
            monkeypatch.setattr(virus_scanner, "_scanner", scanner)
            # Each half would fit under the limit, and neither holds the marker.
            with pytest.raises(VirusScanError, match=error):
                await scan_content_safe(_straddling(3 * _MiB // 2), filename="a.zip")

        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_a_file_within_clamds_stream_limit_is_scanned_whole_in_one_stream(
        self,
    ):
        sessions: list[int] = []
        content = _straddling(3 * _MiB // 2)
        async with _fake_clamd(
            stream_max=2 * _MiB, found_marker=_MARKER, sessions=sessions
        ) as port:
            scanner = VirusScannerService(
                VirusScannerSettings(clamav_service_port=port)
            )
            result = await scanner.scan_file(content)

        assert result.threat_name == "Eicar-Test-Signature"
        assert sessions == [len(content)]

    @pytest.mark.asyncio
    async def test_clamd_closing_the_connection_fails_the_scan(self):
        sessions: list[int] = []
        async with _fake_clamd(drop=True, sessions=sessions) as port:
            scanner = VirusScannerService(
                VirusScannerSettings(
                    clamav_service_port=port, mark_failed_scans_as_clean=True
                )
            )
            with pytest.raises(RuntimeError, match="was not scanned"):
                await scanner.scan_file(b"\x00" * _MiB)

        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_with_scanning_disabled_a_file_of_any_size_is_accepted_unsent(
        self, monkeypatch
    ):
        sessions: list[int] = []
        async with _fake_clamd(stream_max=_MiB, sessions=sessions) as port:
            scanner = VirusScannerService(
                VirusScannerSettings(
                    clamav_service_port=port, clamav_service_enabled=False
                )
            )
            monkeypatch.setattr(virus_scanner, "_scanner", scanner)
            await scan_content_safe(_straddling(2 * _MiB))

        assert sessions == []

    @pytest.mark.asyncio
    async def test_a_clamd_that_never_answers_ping_fails_the_scan_at_the_timeout(
        self,
    ):
        async with _fake_clamd(stall_ping=True) as port:
            scanner = VirusScannerService(
                VirusScannerSettings(clamav_service_port=port, clamav_service_timeout=1)
            )
            with pytest.raises(aioclamd.ClamdConnectionError, match="did not answer"):
                await asyncio.wait_for(scanner.scan_file(b"some upload"), timeout=10)

    def test_the_configured_timeout_reaches_the_scanner(self, monkeypatch):
        monkeypatch.setattr(virus_scanner, "_scanner", None)
        monkeypatch.setattr(virus_scanner.settings.config, "clamav_service_timeout", 7)

        assert get_virus_scanner().settings.clamav_service_timeout == 7

    def test_no_timeout_is_set_by_default_so_deadlines_grow_with_the_file(self):
        assert Config.model_fields["clamav_service_timeout"].default is None

    @pytest.mark.asyncio
    async def test_a_found_reply_reports_the_threat(self):
        async with _fake_clamd(reply=b"stream: Eicar-Test-Signature FOUND\n") as port:
            scanner = VirusScannerService(
                VirusScannerSettings(clamav_service_port=port)
            )
            result = await scanner.scan_file(b"some upload")

        assert not result.is_clean
        assert result.threat_name == "Eicar-Test-Signature"

    @pytest.mark.asyncio
    async def test_a_file_clamd_stops_scanning_at_a_limit_is_refused_as_too_complex(
        self, monkeypatch
    ):
        reply = b"stream: Heuristics.Limits.Exceeded.MaxScanTime FOUND\n"
        async with _fake_clamd(reply=reply) as port:
            scanner = VirusScannerService(
                VirusScannerSettings(
                    clamav_service_port=port, mark_failed_scans_as_clean=True
                )
            )
            monkeypatch.setattr(virus_scanner, "_scanner", scanner)
            with pytest.raises(VirusDetectedError) as caught:
                await scan_content_safe(b"some upload")

        assert caught.value.threat_name == "Heuristics.Limits.Exceeded.MaxScanTime"
        assert str(caught.value) == (
            "File rejected: too large or complex to scan whole "
            "(ClamAV's MaxScanTime limit)"
        )

    @pytest.mark.asyncio
    async def test_an_error_reply_refuses_the_file(self, monkeypatch):
        async with _fake_clamd(reply=b"stream: Can't allocate memory ERROR\n") as port:
            scanner = VirusScannerService(
                VirusScannerSettings(
                    clamav_service_port=port, mark_failed_scans_as_clean=True
                )
            )
            monkeypatch.setattr(virus_scanner, "_scanner", scanner)
            with pytest.raises(VirusScanError, match="allocate memory"):
                await scan_content_safe(b"some upload")

    @pytest.mark.asyncio
    async def test_an_empty_reply_refuses_the_file(self, monkeypatch):
        async with _fake_clamd(reply=b"") as port:
            scanner = VirusScannerService(
                VirusScannerSettings(
                    clamav_service_port=port, mark_failed_scans_as_clean=True
                )
            )
            monkeypatch.setattr(virus_scanner, "_scanner", scanner)
            with pytest.raises(VirusScanError):
                await scan_content_safe(b"some upload")

    @pytest.mark.asyncio
    async def test_a_clamd_that_stops_reading_fails_the_scan_at_the_timeout(self):
        async with _fake_clamd(stall=True) as port:
            scanner = VirusScannerService(
                VirusScannerSettings(clamav_service_port=port, clamav_service_timeout=1)
            )
            with pytest.raises(aioclamd.ClamdConnectionError, match="did not answer"):
                await asyncio.wait_for(
                    scanner.scan_file(b"\x00" * (16 * 1024 * 1024)), timeout=10
                )

    @pytest.mark.asyncio
    async def test_with_no_timeout_set_a_larger_file_gets_longer_to_scan(
        self, monkeypatch
    ):
        # Scaled down to 0.5 s plus 2 s per MiB, against a clamd that answers
        # 2 s after each stream ends.
        monkeypatch.setattr(virus_scanner, "_DEFAULT_TIMEOUT_S", 0.5)
        monkeypatch.setattr(virus_scanner, "_TIMEOUT_S_PER_MIB", 2.0)
        async with _fake_clamd(reply_delay=2.0) as port:
            scanner = VirusScannerService(
                VirusScannerSettings(clamav_service_port=port)
            )
            assert (await scanner.scan_file(bytes(4 * _MiB))).is_clean
            with pytest.raises(aioclamd.ClamdConnectionError, match="did not answer"):
                await scanner.scan_file(b"\x00")

    @pytest.mark.asyncio
    async def test_a_configured_timeout_holds_for_a_large_file(self, monkeypatch):
        monkeypatch.setattr(virus_scanner, "_DEFAULT_TIMEOUT_S", 0.5)
        monkeypatch.setattr(virus_scanner, "_TIMEOUT_S_PER_MIB", 2.0)
        async with _fake_clamd(reply_delay=2.0) as port:
            scanner = VirusScannerService(
                VirusScannerSettings(clamav_service_port=port, clamav_service_timeout=1)
            )
            with pytest.raises(aioclamd.ClamdConnectionError, match="within 1s"):
                await scanner.scan_file(bytes(4 * _MiB))


@pytest.mark.parametrize(
    "compose_file", ["docker-compose.yml", "backend/docker-compose.test.yaml"]
)
def test_the_shipped_clamd_takes_the_largest_upload_whole(compose_file):
    largest_upload_mb = max(
        Config.model_fields[name].default
        for name in ("upload_file_size_limit_mb", "max_file_size_mb")
    )
    limits = dict(
        re.findall(
            r"CLAMD_CONF_(StreamMaxLength|MaxFileSize|MaxScanSize|PCREMaxFileSize)"
            r"=(\d+)M",
            (_PLATFORM_DIR / compose_file).read_text(encoding="utf-8"),
        )
    )

    assert limits.keys() == {
        "StreamMaxLength",
        "MaxFileSize",
        "MaxScanSize",
        "PCREMaxFileSize",
    }
    assert all(int(mb) >= largest_upload_mb for mb in limits.values())


@pytest.mark.parametrize(
    "compose_file", ["docker-compose.yml", "backend/docker-compose.test.yaml"]
)
def test_the_shipped_clamd_flags_a_file_it_stops_scanning_at_a_limit(compose_file):
    text = (_PLATFORM_DIR / compose_file).read_text(encoding="utf-8")
    largest_upload_mb = max(
        Config.model_fields[name].default
        for name in ("upload_file_size_limit_mb", "max_file_size_mb")
    )
    max_scan_time = re.search(r"CLAMD_CONF_MaxScanTime=(\d+)", text)

    assert "CLAMD_CONF_AlertExceedsMax=yes" in text
    assert max_scan_time is not None
    # clamd's own limit, not the backend's deadline, ends the largest scan.
    assert int(max_scan_time[1]) / 1000 < (
        virus_scanner._DEFAULT_TIMEOUT_S
        + virus_scanner._TIMEOUT_S_PER_MIB * largest_upload_mb
    )


@pytest.mark.parametrize("value", ["0", "-5"])
@pytest.mark.parametrize("settings_class", [Config, VirusScannerSettings])
def test_a_clamav_timeout_of_zero_or_less_is_refused(
    monkeypatch, settings_class, value
):
    monkeypatch.setenv("CLAMAV_SERVICE_TIMEOUT", value)

    with pytest.raises(ValidationError, match="clamav_service_timeout"):
        settings_class()


_PLATFORM_DIR = Path(__file__).parents[3]
_MiB = 1024 * 1024
_MARKER = b"synthetic-test-signature"


def _straddling(size: int) -> bytes:
    """`size` zero bytes with `_MARKER` across the middle, so scanning the file
    in halves would cut it in two."""
    start = size // 2 - len(_MARKER) // 2
    return bytes(start) + _MARKER + bytes(size - start - len(_MARKER))


@contextlib.asynccontextmanager
async def _fake_clamd(
    stream_max: int | None = None,
    reply: bytes = b"stream: OK\n",
    stall: bool = False,
    stall_ping: bool = False,
    drop: bool = False,
    read_past_limit: bool = False,
    found_marker: bytes | None = None,
    sessions: list[int] | None = None,
    reply_delay: float = 0.0,
):
    """A ClamAV stand-in. Past `stream_max` it refuses the stream and closes the
    connection, as clamd does, or with `read_past_limit` reads the rest first;
    it answers FOUND for `found_marker` only where one stream holds the whole
    of it, answers `reply_delay` seconds after a stream ends, and logs the
    bytes each INSTREAM session sent into `sessions`."""
    released = asyncio.Event()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        if stall_ping:
            await released.wait()
        elif await reader.readline() == b"nPING\n":
            writer.write(b"PONG\n")
        elif stall:
            await released.wait()
        elif drop:
            if sessions is not None:
                sessions.append(0)
            writer.transport.abort()
            return
        else:
            received = 0
            stream = bytearray()
            refused = False
            while size := struct.unpack("!L", await reader.readexactly(4))[0]:
                frame = await reader.readexactly(size)
                received += len(frame)
                if found_marker is not None:
                    stream += frame
                if stream_max is not None and received > stream_max and not refused:
                    refused = True
                    writer.write(b"INSTREAM size limit exceeded. ERROR\n")
                    if not read_past_limit:
                        break
            if sessions is not None:
                sessions.append(received)
            if reply_delay and not refused:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(released.wait(), reply_delay)
                if released.is_set():  # the client gave up and the test ended
                    writer.close()
                    return
            if not refused:
                found = found_marker is not None and found_marker in stream
                writer.write(
                    b"stream: Eicar-Test-Signature FOUND\n" if found else reply
                )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        released.set()
        server.close()
        await server.wait_closed()


async def _with_longest_stall(coro):
    stalls = [0.0]

    async def watch():
        while True:
            before = time.monotonic()
            await asyncio.sleep(0.01)
            stalls.append(time.monotonic() - before)

    watcher = asyncio.create_task(watch())
    try:
        return await coro, max(stalls)
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
