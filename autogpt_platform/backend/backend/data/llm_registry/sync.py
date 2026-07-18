"""Remote LLM catalog sync for self-hosted installs.

Polls the cloud's public catalog endpoint on startup and daily, importing
the payload through the same idempotent importer the bundled catalog uses.

Safety properties:
- Never runs on cloud deployments (``behave_as == CLOUD``) — cloud DBs are
  admin-managed truth and must never fetch from themselves.
- The payload is fully pydantic-validated (schema version, field bounds,
  size caps) BEFORE any DB write; a malformed or hostile payload is
  rejected wholesale and the last-known-good catalog stays in place.
- The loop never dies: every attempt is wrapped, failures log a warning
  and update ``LlmCatalogState.lastRemoteSyncAt`` for observability.
- A single-key Redis NX lock keeps multi-pod installs from fanning out
  redundant fetches (the import itself is idempotent regardless).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import aiohttp
import prisma.models
from prisma.enums import LlmCatalogSource

from backend.data.llm_registry.catalog_model import CatalogPayload
from backend.data.llm_registry.importer import import_catalog
from backend.data.redis_client import get_redis_async
from backend.util.request import Requests
from backend.util.settings import BehaveAs, Config, Settings

logger = logging.getLogger(__name__)

config = Config()
settings = Settings()

_FETCH_TIMEOUT_SECONDS = 15
_MAX_PAYLOAD_BYTES = 2 * 1024 * 1024  # 2 MiB
_SYNC_LOCK_KEY = "llm_catalog:sync_lock"
_SYNC_LOCK_TTL_SECONDS = 300


def should_sync() -> bool:
    """Sync only on self-hosted (behave_as=LOCAL) installs with the flag on."""
    return (
        config.llm_catalog_sync_enabled and settings.config.behave_as == BehaveAs.LOCAL
    )


async def _acquire_sync_lock() -> bool:
    """Best-effort multi-pod dedup; fail-open (import is idempotent anyway)."""
    try:
        redis = await get_redis_async()
        return bool(
            await redis.set(_SYNC_LOCK_KEY, "1", nx=True, ex=_SYNC_LOCK_TTL_SECONDS)
        )
    except Exception:
        logger.warning("Catalog sync lock unavailable — proceeding", exc_info=True)
        return True


async def _record_attempt(success: bool) -> None:
    now = datetime.now(timezone.utc)
    data: dict = {"lastRemoteSyncAt": now}
    if success:
        data["lastRemoteSuccessAt"] = now
    try:
        await prisma.models.LlmCatalogState.prisma().update_many(
            where={"id": "singleton"}, data=data
        )
    except Exception:
        logger.warning("Failed to record catalog sync attempt", exc_info=True)


async def sync_catalog_once() -> bool:
    """Fetch + validate + import the remote catalog. Returns success."""
    url = config.llm_catalog_url
    response = await Requests(trusted_origins=[url], raise_for_status=True).get(
        url, timeout=aiohttp.ClientTimeout(total=_FETCH_TIMEOUT_SECONDS)
    )

    if len(response.content) > _MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"catalog payload too large: {len(response.content)} bytes "
            f"(max {_MAX_PAYLOAD_BYTES})"
        )

    # Full validation before any DB write — see module docstring.
    payload = CatalogPayload.model_validate_json(response.content)
    result = await import_catalog(
        payload, source=LlmCatalogSource.REMOTE, source_url=url
    )
    logger.info(
        "Remote LLM catalog sync complete (hash=%s, unchanged=%s)",
        result.content_hash[:12],
        result.unchanged,
    )
    return True


async def _sync_once_safe() -> None:
    if not await _acquire_sync_lock():
        logger.debug("Catalog sync lock held elsewhere — skipping this cycle")
        return
    success = False
    try:
        success = await sync_catalog_once()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning(
            "Remote LLM catalog sync failed — keeping last-known-good catalog",
            exc_info=True,
        )
    await _record_attempt(success)


async def llm_catalog_sync_loop() -> None:
    """Long-lived background task: sync at startup, then daily."""
    if not should_sync():
        logger.debug("Remote LLM catalog sync disabled (config or behave_as)")
        return
    await _sync_once_safe()
    while True:
        await asyncio.sleep(config.llm_catalog_sync_interval_hours * 3600)
        await _sync_once_safe()
