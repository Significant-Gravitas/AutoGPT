"""What Sentry would be sent, for tests of what a request leaves behind. The
fixtures built on it are in this directory's conftest.py; ``serve``, which
runs a service's app as its entrypoint does, is in request_url_logging_test.py,
since a module that starts a uvicorn server outside a test is a server the
backend runs (api/middleware/body_limit_test.py)."""

from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport


class CapturingTransport(Transport):
    """Keeps every envelope the SDK sends: ``payload`` as sent, and the
    ``events`` in it."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []
        self.payload = ""

    def capture_envelope(self, envelope: Envelope) -> None:
        self.payload += envelope.serialize().decode()
        self.events += [item.payload.json or {} for item in envelope.items]
