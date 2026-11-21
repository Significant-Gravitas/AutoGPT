"""Reading an E2B sandbox's output and files with a size limit applied while reading."""

from collections.abc import AsyncIterator
from typing import Any, TypeVar

import httpx
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from e2b import AsyncSandbox, CommandResult, ConnectionConfig
from e2b.api import proxy_to_config
from e2b.api.client_async import get_pyqwest_transport
from e2b.envd.api import ENVD_API_FILES_ROUTE
from e2b.envd.client_async import PlainHTTPErrorTransport
from e2b_code_interpreter import AsyncSandbox as AsyncCodeSandbox
from pyqwest import Client, Request, Response

# For one command or one run_code call: its streams and results together.
MAX_SANDBOX_OUTPUT_SIZE = 10 * 1024 * 1024
# For one streamed answer of the box's daemon, whatever its messages hold: output
# is base64 in JSON, a third longer, and the rest is room for the framing.
MAX_SANDBOX_RESPONSE_SIZE = 2 * MAX_SANDBOX_OUTPUT_SIZE
# For one message, and any answer that is not a stream: the SDK buffers a
# message whole and decodes it in one go. The daemon sends output in 32 KiB.
MAX_SANDBOX_MESSAGE_SIZE = 1024 * 1024
_STREAM_CONTENT_TYPE = "application/connect+"
_ENVELOPE_HEADER_SIZE = 5

S = TypeVar("S", bound=AsyncSandbox)


class SandboxOutputLimitError(ValueError):
    """A sandbox sent back more than the limit on what was being read."""


async def run_sandbox_command(
    sandbox: AsyncSandbox, command: str, **kwargs: Any
) -> CommandResult:
    """``sandbox.commands.run`` in the foreground, given up on past the limit
    with ``SandboxOutputLimitError``: a callback that raises ends the SDK's
    reading of the stream. Other keywords go to ``commands.run``."""
    limit = _OutputLimit()
    return await sandbox.commands.run(
        command, on_stdout=limit, on_stderr=limit, **kwargs
    )


async def read_sandbox_file(
    sandbox: AsyncSandbox, path: str, max_bytes: int, **kwargs: Any
) -> bytes:
    """At most ``max_bytes + 1`` bytes of *path*: a result longer than
    *max_bytes* means the file is over the limit. Other keywords go to
    ``files.read``."""
    data = bytearray()
    stream = await sandbox.files.read(path, format="stream", **kwargs)
    async with stream:
        async for chunk in stream:
            data += chunk[: max_bytes + 1 - len(data)]
            if len(data) > max_bytes:
                break
    return bytes(data)


def limit_sandbox_responses(sandbox: S) -> S:
    """Limit every answer of the daemon in *sandbox* while it is read: its RPCs
    (commands, PTYs, file listings) and its file API, all but a file itself,
    which ``read_sandbox_file`` limits. Returns *sandbox*.

    The daemon runs in the box, so it is trusted no more than a command's
    output, while the SDK reads its answers without a limit of its own.
    """
    http_client = Client(
        PlainHTTPErrorTransport(_LimitedTransport(sandbox.connection_config))
    )
    services = (sandbox.commands, sandbox.files, sandbox.pty)
    for service in services:
        # The SDK takes no transport from its caller.
        service._rpc._http_client = http_client
    file_apis = [sandbox._envd_api, sandbox.files._envd_api_streaming]
    # Each service also probes the daemon's health after a failed request.
    file_apis += [service._envd_health_api for service in services]
    for api in file_apis:
        # Uncompressed, so the counter sees what the SDK will decode.
        api.headers["accept-encoding"] = "identity"
        api.event_hooks["response"].append(_count_api_body)
    return sandbox


class OutputLimitedCodeSandbox(AsyncCodeSandbox):
    """The code interpreter sandbox, with ``run_code`` given up on past the limit.

    The response body is counted as it arrives, rather than the messages
    ``run_code`` hands its callbacks, which arrive only once whole. The body is
    asked for uncompressed, because httpx decodes above the stream the counter
    wraps: a compressed body would decode past the limit unmetered.
    ``run_code`` builds a fresh client from ``_client`` for each request.
    """

    @property
    def _client(self) -> httpx.AsyncClient:
        client = super()._client
        client.headers["accept-encoding"] = "identity"
        client.event_hooks["response"].append(_count_body)
        return client


class _OutputLimit:
    def __init__(self, limit: int = MAX_SANDBOX_OUTPUT_SIZE) -> None:
        self._limit = limit
        self._size = 0

    def __call__(self, output: str | bytes) -> None:
        self._size += len(output)
        if self._size > self._limit:
            raise SandboxOutputLimitError(
                f"Sandbox output exceeds the limit of {self._limit // 2**20} MiB"
            )


class _LimitedTransport:
    """The SDK's transport for a sandbox's RPCs, with bodies cut off at the
    limits by a ``ConnectError``, which connectrpc raises as it is."""

    def __init__(self, config: ConnectionConfig) -> None:
        self._config = config

    async def execute(self, request: Request) -> Response:
        # The cached pool the SDK's own RPC clients use (`create_rpc_client`).
        pool = get_pyqwest_transport(
            proxy_to_config(self._config.proxy), http_version=self._config.http_version
        )
        response = await pool.execute(request)
        return Response(
            status=response.status,
            http_version=response.http_version,
            headers=response.headers,
            content=_limited_body(response),
            trailers=response.trailers,
        )


async def _limited_body(
    response: Response,
) -> AsyncIterator[bytes | bytearray | memoryview]:
    content_type = response.headers.get("content-type", "")
    streamed = response.status == 200 and content_type.startswith(_STREAM_CONTENT_TYPE)
    limit = MAX_SANDBOX_RESPONSE_SIZE if streamed else MAX_SANDBOX_MESSAGE_SIZE
    size = 0
    header = b""
    # Bytes of the current message still to come.
    pending = 0
    try:
        async for chunk in response.content:
            size += len(chunk)
            if size > limit:
                raise _over_the_limit("response", limit)
            offset = 0
            while streamed and offset < len(chunk):
                if pending:
                    passed = min(pending, len(chunk) - offset)
                    pending -= passed
                    offset += passed
                    continue
                part = bytes(
                    chunk[offset : offset + _ENVELOPE_HEADER_SIZE - len(header)]
                )
                header += part
                offset += len(part)
                if len(header) == _ENVELOPE_HEADER_SIZE:
                    # A flags byte, then the length: refused before it is buffered.
                    pending = int.from_bytes(header[1:], "big")
                    header = b""
                    if pending > MAX_SANDBOX_MESSAGE_SIZE:
                        raise _over_the_limit("message", MAX_SANDBOX_MESSAGE_SIZE)
            yield chunk
    finally:
        # The SDK relies on a stream it stops reading early being reset.
        await response.aclose()


def _over_the_limit(what: str, limit: int) -> ConnectError:
    # INTERNAL is the code the SDK gives an answer it cannot decode.
    return ConnectError(
        Code.INTERNAL, f"Sandbox sent a {what} over the limit of {limit // 2**20} MiB"
    )


async def _count_api_body(response: httpx.Response) -> None:
    # Everything but a file the SDK reads whole, and decodes if it is JSON.
    request = response.request
    is_file = (
        request.method == "GET"
        and request.url.path.endswith(ENVD_API_FILES_ROUTE)
        and response.is_success
    )
    if not is_file:
        await _count_body(response, MAX_SANDBOX_MESSAGE_SIZE)


async def _count_body(
    response: httpx.Response, limit: int = MAX_SANDBOX_OUTPUT_SIZE
) -> None:
    if not isinstance(response.stream, httpx.AsyncByteStream):
        raise TypeError("Expected an async response stream")
    # The counter sees the bytes on the wire, so an encoded body would decode
    # past the limit unmetered however small it arrives.
    encoding = response.headers.get("content-encoding", "identity").lower()
    if encoding != "identity":
        raise SandboxOutputLimitError(
            f"Sandbox sent a {encoding}-encoded body, which cannot be limited "
            "while reading"
        )
    response.stream = _CountedBody(response.stream, limit)


class _CountedBody(httpx.AsyncByteStream):
    def __init__(self, body: httpx.AsyncByteStream, limit: int) -> None:
        self._body = body
        self._limit = limit

    async def __aiter__(self) -> AsyncIterator[bytes]:
        limit = _OutputLimit(self._limit)
        try:
            async for chunk in self._body:
                limit(chunk)
                yield chunk
        except SandboxOutputLimitError:
            # The SDK leaves open a response whose error it could not read.
            await self.aclose()
            raise

    async def aclose(self) -> None:
        await self._body.aclose()
