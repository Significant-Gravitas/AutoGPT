import asyncio
import io
import logging
import struct
import time
import warnings
from typing import BinaryIO, Optional, Tuple

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


class VirusScanResult(BaseModel):
    is_clean: bool
    scan_time_ms: int
    file_size: int
    threat_name: Optional[str] = None


class VirusScannerSettings(BaseSettings):
    # Tunables for the scanner layer (NOT the ClamAV daemon).
    clamav_service_host: str = "localhost"
    clamav_service_port: int = 3310
    clamav_service_timeout: int = 60
    clamav_service_enabled: bool = True
    # If the service is disabled, all files are considered clean.
    mark_failed_scans_as_clean: bool = False
    # Client-side protective limits
    max_scan_size: int = 2 * 1024 * 1024 * 1024  # 2 GB guard-rail in memory
    min_chunk_size: int = 128 * 1024  # 128 KB hard floor
    max_retries: int = 8  # halve ≤ max_retries times
    # Concurrency throttle toward the ClamAV daemon.  Do *NOT* simply turn this
    # up to the number of CPU cores; keep it ≤ (MaxThreads / pods) – 1.
    max_concurrency: int = 5


class VirusScannerService:
    """Fully-async ClamAV wrapper using **aioclamd**.

    • Reuses a single `ClamdAsyncClient` connection (aioclamd keeps the socket open).
    • Throttles concurrent `INSTREAM` calls with an `asyncio.Semaphore` so we don't exhaust daemon worker threads or file descriptors.
    • Falls back to progressively smaller chunk sizes when the daemon rejects a stream as too large.
    """

    def __init__(self, settings: VirusScannerSettings) -> None:
        self.settings = settings
        self._client = _DrainingClamdClient(
            host=settings.clamav_service_host,
            port=settings.clamav_service_port,
            timeout=settings.clamav_service_timeout,
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
            return False, None
        status, threat = next(iter(raw.values()))
        return status == "FOUND", threat

    async def _instream(self, chunk: bytes) -> Tuple[bool, Optional[str]]:
        """Scan **one** chunk with concurrency control."""
        async with self._sem:
            try:
                raw = await self._client.instream(io.BytesIO(chunk))
                return self._parse_raw(raw)
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise RuntimeError("size-limit") from exc
            except Exception as exc:
                if "INSTREAM size limit exceeded" in str(exc):
                    raise RuntimeError("size-limit") from exc
                raise

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def scan_file(
        self, content: bytes, *, filename: str = "unknown"
    ) -> VirusScanResult:
        """
        Scan `content`.  Returns a result object or raises on infrastructure
        failure (unreachable daemon, etc.).
        The algorithm always tries whole-file first. If the daemon refuses
        on size grounds, it falls back to chunked parallel scanning.
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
        chunk_size = max(1, len(content))  # Start with full content length
        for retry in range(self.settings.max_retries):
            # For small files, don't check min_chunk_size limit
            if chunk_size < self.settings.min_chunk_size and chunk_size < len(content):
                break
            logger.debug(
                f"Scanning {filename} with chunk size: {chunk_size // 1_048_576} MB (retry {retry + 1}/{self.settings.max_retries})"
            )
            tasks = [
                asyncio.create_task(self._instream(content[o : o + chunk_size]))
                for o in range(0, len(content), chunk_size)
            ]
            try:
                for coro in asyncio.as_completed(tasks):
                    infected, threat = await coro
                    if infected:
                        return VirusScanResult(
                            is_clean=False,
                            threat_name=threat,
                            file_size=len(content),
                            scan_time_ms=int((time.monotonic() - start) * 1000),
                        )
                # All chunks clean
                return VirusScanResult(
                    is_clean=True,
                    file_size=len(content),
                    scan_time_ms=int((time.monotonic() - start) * 1000),
                )
            except RuntimeError as exc:
                if str(exc) == "size-limit":
                    chunk_size //= 2
                    continue
                logger.error(f"Cannot scan {filename}: {exc}")
                raise
            finally:
                # A chunk left running would keep streaming to ClamAV after the scan ended.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        # Phase 3 – give up but warn
        logger.warning(
            f"Unable to virus scan {filename} ({len(content)} bytes) even with minimum chunk size ({self.settings.min_chunk_size} bytes). Recommend manual review."
        )
        return VirusScanResult(
            is_clean=self.settings.mark_failed_scans_as_clean,
            file_size=len(content),
            scan_time_ms=int((time.monotonic() - start) * 1000),
        )


class _DrainingClamdClient(aioclamd.ClamdAsyncClient):
    """aioclamd 1.0.0's ``instream`` ignores ``timeout`` and queues the whole buffer
    unyielding, quadratic on Python 3.12+: it held a copilot turn's loop for minutes."""

    async def instream(self, buffer: BinaryIO) -> dict:
        try:
            return await asyncio.wait_for(self._stream(buffer), self.timeout)
        except TimeoutError:
            raise aioclamd.ClamdConnectionError(
                f"ClamAV did not answer within {self.timeout}s"
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
        finally:
            writer.close()
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
