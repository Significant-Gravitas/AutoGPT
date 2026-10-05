import time

from pydantic import BaseModel


class NetworkMetadata(BaseModel):
    requestId: str = ""
    type: str = ""


class NetworkDrain:
    def __init__(self):
        self.pending: set[tuple[str, str]] = set()
        self.started = time.monotonic()
        self.changed = self.started

    def observe(self, method: str, session: str, metadata: NetworkMetadata) -> None:
        key = (session, metadata.requestId)
        if method == "Network.requestWillBeSent" and metadata.type != "WebSocket":
            if len(self.pending) >= 1000:
                raise RuntimeError("Checkout network activity exceeds its bound")
            self.pending.add(key)
            self.changed = time.monotonic()
        elif method in {"Network.loadingFinished", "Network.loadingFailed"}:
            self.pending.discard(key)
            self.changed = time.monotonic()

    def settled(self) -> bool:
        now = time.monotonic()
        return not self.pending and now - self.started >= 3 and now - self.changed >= 1

    def start_wait(self) -> None:
        self.started = time.monotonic()
