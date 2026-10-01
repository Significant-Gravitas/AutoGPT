"""Cross-process broadcast of credential changes.

The API server and the copilot executor are separate processes and each holds
its own credential caches.  A write serves one request in one of them, so an
in-process callback only ever reaches that process's copy; every write is also
published here so the other processes drop theirs.
"""

from collections.abc import AsyncGenerator

from pydantic import BaseModel

from backend.data.event_bus import AsyncRedisEventBus

# Sharded pub/sub has no pattern-subscribe, and a subscriber cannot know which
# users it holds tokens for — so everything goes on one channel.
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
