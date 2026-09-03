"""Cross-process broadcast of credential changes.

A credential write happens in whichever process serves the request — the API
process for an OAuth callback — while the caches holding the stale token live
in other processes, such as the copilot executor.  An in-process callback
cannot cross that boundary, so every write is also published here and picked
up by subscribers wherever they run.
"""

import logging
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from backend.data.event_bus import AsyncRedisEventBus

logger = logging.getLogger(__name__)

# Sharded pub/sub has no pattern-subscribe and a subscriber cannot enumerate
# the users whose tokens it may be holding, so every change goes on one
# broadcast channel and subscribers filter locally.
CREDS_CHANGED_CHANNEL = "all"


class CredentialsChangedEvent(BaseModel):
    user_id: str
    provider: str


async def publish_creds_changed(user_id: str, provider: str) -> None:
    """Announce that *user_id*'s *provider* credentials were written."""
    await _bus.publish_event(
        CredentialsChangedEvent(user_id=user_id, provider=provider),
        CREDS_CHANGED_CHANNEL,
    )


async def listen_creds_changed() -> AsyncGenerator[CredentialsChangedEvent, None]:
    """Yield credential-change events published by any process."""
    async for event in _bus.listen_events(CREDS_CHANGED_CHANNEL):
        yield event


class CredentialsChangedEventBus(AsyncRedisEventBus[CredentialsChangedEvent]):
    Model = CredentialsChangedEvent

    @property
    def event_bus_name(self) -> str:
        return "creds_changed"


_bus = CredentialsChangedEventBus()
