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
    held back until it ends or passes ``max_size``, so at most ``max_size``
    bytes are buffered: one that ends within the cap is compressed chunk by
    chunk, and one that passes it is sent uncompressed.
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
        self.length_unknown = False
        self.pending: list[Message] = []
        self.pending_size = 0

    async def send_with_compression(self, message: Message) -> None:
        if self.passthrough:
            await self.send(message)
            return
        if message["type"] == "http.response.start":
            declared_length = _declared_length(message)
            if declared_length > self.max_size:
                self.passthrough = True
                await self.send(message)
                return
            self.length_unknown = declared_length < 0
            await super().send_with_compression(message)
            return
        if (
            message["type"] != "http.response.body"
            or self.started
            or self.content_encoding_set
            or self.content_type_is_excluded
        ):
            await super().send_with_compression(message)
            return
        if not self.length_unknown:
            if len(message.get("body", b"")) > self.max_size:
                await self._send_uncompressed([message])
                return
            await super().send_with_compression(message)
            return
        self.pending.append(message)
        self.pending_size += len(message.get("body", b""))
        if self.pending_size > self.max_size:
            await self._send_uncompressed(self.pending)
            return
        if message.get("more_body", False):
            return
        pending, self.pending = self.pending, []
        for pending_message in pending:
            await super().send_with_compression(pending_message)

    async def _send_uncompressed(self, messages: list[Message]) -> None:
        self.passthrough = True
        self.started = True
        self.pending = []
        await self.send(self.initial_message)
        for message in messages:
            await self.send(message)


def _declared_length(start_message: Message) -> int:
    """The response's ``Content-Length``, or -1 when it has none."""
    value = Headers(raw=start_message["headers"]).get("content-length", "")
    return int(value) if value.isdigit() else -1
