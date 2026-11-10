import asyncio
import contextlib
import io
import logging
import struct
import time
import warnings
from collections.abc import Awaitable
from typing import BinaryIO, Optional, Tuple, TypeVar

# Suppress the specific pkg_resources deprecation warning from aioclamd
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="pkg_resources is deprecated", category=UserWarning
    )
    import aioclamd

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

_STREAM_FRAME_SIZE = 64 * 1024
_MiB = 1024 * 1024
# With no timeout configured, a scan gets this long plus this much per MiB of
# the file, so a large upload has time to stream to ClamAV and scan whole.
_DEFAULT_TIMEOUT_S = 60.0
_TIMEOUT_S_PER_MIB = 1.0

T = TypeVar("T")


class VirusScanResult(BaseModel):
    is_clean: bool
    scan_time_ms: int
    file_size: int
    threat_name: Optional[str] = None


class VirusScannerSettings(BaseSettings):
    # Tunables for the scanner layer (NOT the ClamAV daemon).
    clamav_service_host: str = "localhost"
    clamav_service_port: int = 3310
    # Seconds to wait for ClamAV; None scales each scan's deadline with the file.
    clamav_service_timeout: Optional[int] = None
    clamav_service_enabled: bool = True
    # If the service is disabled, all files are considered clean.
    mark_failed_scans_as_clean: bool = False
    # Client-side protective limits
    max_scan_size: int = 2 * 1024 * 1024 * 1024  # 2 GB guard-rail in memory
    # Concurrency throttle toward the ClamAV daemon.  Do *NOT* simply turn this
    # up to the number of CPU cores; keep it ≤ (MaxThreads / pods) – 1.
    max_concurrency: int = 5


class VirusScannerService:
    """Fully-async ClamAV wrapper using **aioclamd**.

    • Streams each file whole through one connection per `INSTREAM` call.
    • Throttles concurrent `INSTREAM` calls with an `asyncio.Semaphore` so we don't exhaust daemon worker threads or file descriptors.
    • Fails the scan when the daemon won't take the whole file (one over its StreamMaxLength) instead of scanning it in pieces, where a signature spanning a cut, or one hashing the whole file, can never match.
    """

    def __init__(self, settings: VirusScannerSettings) -> None:
        self.settings = settings
        self._client = _DrainingClamdClient(
            host=settings.clamav_service_host,
            port=settings.clamav_service_port,
            timeout=settings.clamav_service_timeout or _DEFAULT_TIMEOUT_S,
        )
        self._sem = asyncio.Semaphore(settings.max_concurrency)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_raw(raw: Optional[dict]) -> Tuple[bool, Optional[str]]:
        """
        Convert aioclamd output to (infected?, threat_name).
        Returns (False, None) for clean.
        """
        if not raw:
            raise aioclamd.ResponseError("ClamAV returned no result")
        status, threat = next(iter(raw.values()))
        if status == "ERROR":
            raise aioclamd.ResponseError(f"ClamAV could not scan the stream: {threat}")
        return status == "FOUND", threat

    async def _instream(
        self, content: bytes, filename: str
    ) -> Tuple[bool, Optional[str]]:
        """Scan all of `content` in one stream, with concurrency control."""
        size = f"{len(content) / _MiB:.1f} MB"
        timeout = self.settings.clamav_service_timeout or (
            _DEFAULT_TIMEOUT_S + _TIMEOUT_S_PER_MIB * len(content) / _MiB
        )
        async with self._sem:
            try:
                raw = await self._client.instream(io.BytesIO(content), timeout)
                return self._parse_raw(raw)
            except (
                BrokenPipeError,
                ConnectionAbortedError,
                ConnectionResetError,
            ) as exc:
                raise RuntimeError(
                    f"{filename} ({size}) was not scanned: ClamAV closed the "
                    "connection before answering, which it does to a stream over "
                    "its StreamMaxLength"
                ) from exc
            except Exception as exc:
                if "INSTREAM size limit exceeded" in str(exc):
                    raise RuntimeError(
                        f"{filename} ({size}) is too large to scan: it is over "
                        "ClamAV's StreamMaxLength"
                    ) from exc
                raise

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def scan_file(
        self, content: bytes, *, filename: str = "unknown"
    ) -> VirusScanResult:
        """
        Scan `content` whole, in one stream.  Returns a result object or raises
        on infrastructure failure (unreachable daemon, etc.), and when the
        daemon won't take the whole file: a file over its StreamMaxLength is
        refused, whatever `mark_failed_scans_as_clean` says.
        """
        if not self.settings.clamav_service_enabled:
            logger.warning(f"Virus scanning disabled – accepting {filename}")
            return VirusScanResult(
                is_clean=True, scan_time_ms=0, file_size=len(content)
            )
        if len(content) == 0:
            logger.debug(f"Skipping virus scan for empty file {filename}")
            return VirusScanResult(is_clean=True, scan_time_ms=0, file_size=0)
        if len(content) > self.settings.max_scan_size:
            logger.warning(
                f"File {filename} ({len(content)} bytes) exceeds client max scan size ({self.settings.max_scan_size}); Stopping virus scan"
            )
            return VirusScanResult(
                is_clean=self.settings.mark_failed_scans_as_clean,
                file_size=len(content),
                scan_time_ms=0,
            )

        # Ensure daemon is reachable (small RTT check)
        if not await self._client.ping():
            raise RuntimeError("ClamAV service is unreachable")

        start = time.monotonic()
        infected, threat = await self._instream(content, filename)
        return VirusScanResult(
            is_clean=not infected,
            threat_name=threat if infected else None,
            file_size=len(content),
            scan_time_ms=int((time.monotonic() - start) * 1000),
        )


class _DrainingClamdClient(aioclamd.ClamdAsyncClient):
    """aioclamd 1.0.0 ignores ``timeout``, and its ``instream`` queues the whole buffer
    unyielding, quadratic on Python 3.12+: it held a copilot turn's loop for minutes."""

    async def instream(self, buffer: BinaryIO, timeout: Optional[float] = None) -> dict:
        return await self._answered(self._stream(buffer), timeout or self.timeout)

    async def ping(self):
        return await self._answered(super().ping(), self.timeout)

    async def _answered(self, call: Awaitable[T], timeout: float) -> T:
        try:
            return await asyncio.wait_for(call, timeout)
        except TimeoutError:
            raise aioclamd.ClamdConnectionError(
                f"ClamAV did not answer within {timeout:.0f}s"
            ) from None

    async def _stream(self, buffer: BinaryIO) -> dict:
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            writer.write(b"nINSTREAM\n")
            while frame := buffer.read(_STREAM_FRAME_SIZE):
                writer.write(struct.pack("!L", len(frame)) + frame)
                await writer.drain()
            writer.write(struct.pack("!L", 0))
            await writer.drain()
            reply = (await reader.read()).decode().strip()
        except BaseException:
            # Unsent frames would hold the socket, and wait_closed(), until ClamAV reads them.
            writer.transport.abort()
            raise
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
        if reply == "INSTREAM size limit exceeded. ERROR":
            raise aioclamd.BufferTooLongError(reply)
        match = aioclamd.scan_response.match(reply)
        if match is None:
            raise aioclamd.ResponseError(reply or "ClamAV closed without a reply")
        return {match["path"]: (match["status"], match["virus"])}


_scanner: Optional[VirusScannerService] = None


def get_virus_scanner() -> VirusScannerService:
    global _scanner
    if _scanner is None:
        _settings = VirusScannerSettings(
            clamav_service_host=settings.config.clamav_service_host,
            clamav_service_port=settings.config.clamav_service_port,
            clamav_service_timeout=settings.config.clamav_service_timeout,
            clamav_service_enabled=settings.config.clamav_service_enabled,
            max_concurrency=settings.config.clamav_max_concurrency,
            mark_failed_scans_as_clean=settings.config.clamav_mark_failed_scans_as_clean,
        )
        _scanner = VirusScannerService(_settings)
    return _scanner


async def scan_content_safe(content: bytes, *, filename: str = "unknown") -> None:
    """
    Helper function to scan content and raise appropriate exceptions.

    Raises:
        VirusDetectedError: If virus is found
        VirusScanError: If scanning fails
    """
    from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError

    try:
        result = await get_virus_scanner().scan_file(content, filename=filename)
        if not result.is_clean:
            threat_name = result.threat_name or "Unknown threat"
            logger.warning(f"Virus detected in file {filename}: {threat_name}")
            raise VirusDetectedError(
                threat_name, f"File rejected due to virus detection: {threat_name}"
            )

        logger.info(f"File {filename} passed virus scan in {result.scan_time_ms}ms")

    except VirusDetectedError:
        raise
    except Exception as e:
        logger.warning(f"Virus scanning failed for {filename}: {str(e)}")
        raise VirusScanError(f"Virus scanning failed: {str(e)}") from e
