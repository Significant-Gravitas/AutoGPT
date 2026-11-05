"""Say plainly that a service cannot be connected yet, and record the request.

Without this, a search that found nothing ended in prose ("nothing built in,
so it connects through its API") and a hand-built HTTP request asking a new
user for a raw token. Every request for the same service lands on one
feature request, one customer need per ask, so the gap is counted.
"""

import asyncio
import logging
import uuid
from typing import Any

from backend.api.features.integrations.support import canonical_name, find_connectable
from backend.copilot.model import ChatSession
from backend.data.redis_client import get_redis_async
from backend.executor.cluster_lock import AsyncClusterLock

from .base import BaseTool
from .feature_requests import file_feature_request, find_feature_request
from .models import (
    ErrorResponse,
    IntegrationRequestResponse,
    ResponseType,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

TITLE_PREFIX = "Integration request: "
MAX_SERVICE_CHARS = 60
MAX_USE_CASE_CHARS = 500
# Look-up-then-file for one service runs under this lock across workers, so
# two first asks cannot each find nothing and file their own request.
LOCK_KEY_PREFIX = "copilot:integration_request:"
LOCK_TTL_SECONDS = 30
LOCK_ATTEMPTS = 20
LOCK_RETRY_SECONDS = 0.1
# The card is the user's answer and must not wait on Linear: past this the
# ask goes unrecorded and the card says so.
RECORD_TIMEOUT_SECONDS = 10.0


class RequestIntegrationTool(BaseTool):
    @property
    def name(self) -> str:
        return "request_integration"

    @property
    def description(self) -> str:
        return (
            "Not-supported card for a service nothing covers; records the "
            "request. Call before any workaround."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "e.g. 'Shopify'"},
                "use_case": {
                    "type": "string",
                    "description": "What for, briefly. No personal info.",
                },
            },
            "required": ["service"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        service: str = "",
        use_case: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        name = canonical_name(service or "")[:MAX_SERVICE_CHARS]
        if not name:
            return ErrorResponse(
                message="Name the service the user wants.", session_id=session_id
            )
        if not user_id:
            return ErrorResponse(
                message="Authentication required.", session_id=session_id
            )
        supported = find_connectable(name)
        if supported is not None:
            return ErrorResponse(
                message=(
                    f"{supported.name} can be connected: "
                    f"find_capability(query='{supported.key}') lists what it does. "
                    "Do not tell the user it is unsupported."
                ),
                session_id=session_id,
            )
        recorded = await _record(user_id, name, use_case, session_id)
        return IntegrationRequestResponse(
            service=name,
            recorded=recorded,
            message=_message(name, recorded),
            session_id=session_id,
        )


async def _record(
    user_id: str, service: str, use_case: str, session_id: str | None
) -> bool:
    """Add this ask to the service's feature request, filing it if new.

    Anything short of a clean answer in time leaves the ask unrecorded rather
    than filed twice: the card is the user's answer either way.
    """
    try:
        async with asyncio.timeout(RECORD_TIMEOUT_SECONDS):
            return await _record_under_lock(user_id, service, use_case, session_id)
    except TimeoutError:
        logger.warning(f"Integration request for {service} was not recorded: timeout")
        return False


async def _record_under_lock(
    user_id: str, service: str, use_case: str, session_id: str | None
) -> bool:
    lock = await _acquire_service_lock(service)
    if lock is None:
        logger.warning(f"Integration request for {service} was not recorded: busy")
        return False
    try:
        return await _file_once(user_id, service, use_case, session_id)
    finally:
        await _release_service_lock(lock)


async def _file_once(
    user_id: str, service: str, use_case: str, session_id: str | None
) -> bool:
    title = f"{TITLE_PREFIX}{service}"
    try:
        existing = await find_feature_request(title)
    except Exception as e:
        # Not knowing whether the request exists is not knowing it doesn't.
        logger.warning(f"Integration request lookup for {service} failed: {e}")
        return False
    description = (use_case or "").strip()[:MAX_USE_CASE_CHARS]
    filed = await file_feature_request(
        user_id,
        title,
        description or f"A user wants to connect {service}.",
        existing,
        session_id,
    )
    if filed.type == ResponseType.FEATURE_REQUEST_CREATED:
        return True
    logger.warning(f"Integration request for {service} was not recorded")
    return False


async def _acquire_service_lock(service: str) -> AsyncClusterLock | None:
    """The service's filing lock, or ``None`` if it stays held or Redis is
    unreachable."""
    try:
        lock = AsyncClusterLock(
            redis=await get_redis_async(),
            key=f"{LOCK_KEY_PREFIX}{service.casefold()}",
            owner_id=uuid.uuid4().hex,
            timeout=LOCK_TTL_SECONDS,
        )
        for _ in range(LOCK_ATTEMPTS):
            if (await lock.try_acquire()) == lock.owner_id:
                return lock
            await asyncio.sleep(LOCK_RETRY_SECONDS)
    except Exception as e:
        logger.warning(f"Integration request lock for {service} failed: {e}")
    return None


async def _release_service_lock(lock: AsyncClusterLock) -> None:
    try:
        await lock.release()
    except Exception as e:  # it expires on its own
        logger.warning(f"Integration request lock release failed: {e}")


def _message(service: str, recorded: bool) -> str:
    noted = " and that their request is recorded" if recorded else ""
    return (
        f"{service} is not supported yet. The card already tells the user "
        f"so{noted}; do not repeat it. Go straight to what you can do with "
        "the services they can connect. Offer a hand-built HTTP request only "
        f"if they ask for a workaround, and never as a way to sign in to "
        f"{service}."
    )
