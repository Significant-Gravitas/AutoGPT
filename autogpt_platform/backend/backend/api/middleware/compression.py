from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware, GZipResponder
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Responses smaller than this are sent as they are.
GZIP_MINIMUM_SIZE = 50_000
# A level that is cheap per byte, rather than Starlette's default of 9.
GZIP_COMPRESSION_LEVEL = 6
# Responses larger than this are sent uncompressed.
GZIP_MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class BoundedGZipMiddleware(GZipMiddleware):
    """Starlette's GZip middleware with a bounded cost per response.

    A response is left uncompressed when its ``Content-Length``, or its first
    body chunk, is over ``max_size``. A streamed response without a length is
    compressed chunk by chunk, so each step's cost is bounded by its chunk.
    """

    def __init__(
        self,
        app: ASGIApp,
        minimum_size: int = GZIP_MINIMUM_SIZE,
        compresslevel: int = GZIP_COMPRESSION_LEVEL,
        max_size: int = GZIP_MAX_RESPONSE_BYTES,
    ) -> None:
        super().__init__(app, minimum_size=minimum_size, compresslevel=compresslevel)
        self.max_size = max_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or "gzip" not in Headers(scope=scope).get(
            "Accept-Encoding", ""
        ):
            await super().__call__(scope, receive, send)
            return
        responder = _BoundedGZipResponder(
            self.app, self.minimum_size, self.compresslevel, self.max_size
        )
        await responder(scope, receive, send)


class _BoundedGZipResponder(GZipResponder):
    def __init__(
        self, app: ASGIApp, minimum_size: int, compresslevel: int, max_size: int
    ) -> None:
        super().__init__(app, minimum_size, compresslevel=compresslevel)
        self.max_size = max_size
        self.passthrough = False

    async def send_with_compression(self, message: Message) -> None:
        if self.passthrough:
            await self.send(message)
            return
        if message["type"] == "http.response.start" and (
            _declared_length(message) > self.max_size
        ):
            self.passthrough = True
            await self.send(message)
            return
        if (
            message["type"] == "http.response.body"
            and not self.started
            and len(message.get("body", b"")) > self.max_size
        ):
            self.passthrough = True
            self.started = True
            await self.send(self.initial_message)
            await self.send(message)
            return
        await super().send_with_compression(message)


def _declared_length(start_message: Message) -> int:
    """The response's ``Content-Length``, or -1 when it has none."""
    value = Headers(raw=start_message["headers"]).get("content-length", "")
    return int(value) if value.isdigit() else -1
