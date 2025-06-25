import asyncio
import logging
import time
from typing import Optional, Tuple

import pyclamd
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class VirusScanResult(BaseModel):
    is_clean: bool
    scan_time_ms: int
    file_size: int
    threat_name: Optional[str] = None


class VirusScannerSettings(BaseSettings):
    clamav_service_host: str = "localhost"
    clamav_service_port: int = 3310
    clamav_service_timeout: int = 60
    clamav_service_enabled: bool = True
    max_scan_size: int = 100 * 1024 * 1024  # 100 MB
    chunk_size: int = 25 * 1024 * 1024  # 25 MB (safe for 50MB stream limit)
    min_chunk_size: int = 128 * 1024  # 128 KB minimum
    max_retries: int = 8  # halve chunk ≤ max_retries times


class VirusScannerService:
    """
    Thin async wrapper around ClamAV.  Creates a fresh `ClamdNetworkSocket`
    per chunk (the class is *not* thread-safe) and falls back to smaller
    chunks when the daemon rejects the stream size.
    """

    def __init__(self, settings: VirusScannerSettings) -> None:
        self.settings = settings

    def _new_client(self) -> pyclamd.ClamdNetworkSocket:
        return pyclamd.ClamdNetworkSocket(
            host=self.settings.clamav_service_host,
            port=self.settings.clamav_service_port,
            timeout=self.settings.clamav_service_timeout,
        )

    @staticmethod
    def _parse_raw(raw: Optional[dict]) -> Tuple[bool, Optional[str]]:
        """
        Convert pyclamd output to (infected?, threat_name).
        Returns (False, None) for clean.
        """
        if not raw:
            return False, None
        status, threat = next(iter(raw.values()))
        return status == "FOUND", threat

    async def _scan_chunk(self, chunk: bytes) -> Tuple[bool, Optional[str]]:
        loop = asyncio.get_running_loop()
        client = self._new_client()
        try:
            raw = await loop.run_in_executor(None, client.scan_stream, chunk)
            return self._parse_raw(raw)

        # ClamAV aborts the socket when >StreamMaxLength → BrokenPipe/Reset.
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise RuntimeError("size-limit") from exc
        except Exception as exc:
            if "INSTREAM size limit exceeded" in str(exc):
                raise RuntimeError("size-limit") from exc
            raise

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    async def scan_file(
        self, content: bytes, *, filename: str = "unknown"
    ) -> VirusScanResult:
        """
        Scan `content`.  Returns a result object or raises on infrastructure
        failure (unreachable daemon, etc.).
        """
        if not self.settings.clamav_service_enabled:
            logger.warning("Virus scanning disabled – accepting %s", filename)
            return VirusScanResult(
                is_clean=True, scan_time_ms=0, file_size=len(content)
            )

        if len(content) > self.settings.max_scan_size:
            logger.warning(
                f"File {filename} ({len(content)} bytes) exceeds max scan size ({self.settings.max_scan_size}), skipping virus scan"
            )
            return VirusScanResult(
                is_clean=True,  # Assume clean for oversized files
                file_size=len(content),
                scan_time_ms=0,
                threat_name=None,
            )

        loop = asyncio.get_running_loop()
        if not await loop.run_in_executor(None, self._new_client().ping):
            raise RuntimeError("ClamAV service is unreachable")

        start = time.monotonic()
        chunk_size = self.settings.chunk_size

        for retry in range(self.settings.max_retries + 1):
            try:
                logger.debug(
                    f"Scanning {filename} with chunk size: {chunk_size // 1_048_576}MB"
                )

                # Scan all chunks with current chunk size
                for offset in range(0, len(content), chunk_size):
                    chunk_data = content[offset : offset + chunk_size]
                    infected, threat = await self._scan_chunk(chunk_data)
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
                if (
                    str(exc) == "size-limit"
                    and chunk_size > self.settings.min_chunk_size
                ):
                    chunk_size //= 2
                    logger.info(
                        f"Chunk size too large for {filename}, reducing to {chunk_size // 1_048_576}MB (retry {retry + 1}/{self.settings.max_retries + 1})"
                    )
                    continue
                else:
                    # Either not a size-limit error, or we've hit minimum chunk size
                    logger.error(f"Cannot scan {filename}: {exc}")
                    raise

        # If we can't scan even with minimum chunk size, log warning and allow file
        logger.warning(
            f"Unable to virus scan {filename} ({len(content)} bytes) - chunk size limits exceeded. "
            f"Allowing file but recommend manual review."
        )
        return VirusScanResult(
            is_clean=True,  # Allow file when scanning impossible
            file_size=len(content),
            scan_time_ms=int((time.monotonic() - start) * 1000),
            threat_name=None,
        )


_scanner: Optional[VirusScannerService] = None


def get_virus_scanner() -> VirusScannerService:
    global _scanner
    if _scanner is None:
        _settings = VirusScannerSettings(
            clamav_service_host=settings.config.clamav_service_host,
            clamav_service_port=settings.config.clamav_service_port,
            clamav_service_enabled=settings.config.clamav_service_enabled,
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
    from backend.server.v2.store.exceptions import VirusDetectedError, VirusScanError

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
        logger.error(f"Virus scanning failed for {filename}: {str(e)}")
        raise VirusScanError(f"Virus scanning failed: {str(e)}") from e
