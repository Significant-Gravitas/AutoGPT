"""Reading an E2B sandbox's output and files with a size limit applied while reading."""

from collections.abc import AsyncIterator
from typing import Any

import httpx
from e2b import AsyncSandbox, CommandResult
from e2b_code_interpreter import AsyncSandbox as AsyncCodeSandbox

# For one command or one run_code call: its streams and results together.
MAX_SANDBOX_OUTPUT_SIZE = 10 * 1024 * 1024


class SandboxOutputLimitError(ValueError):
    """A sandbox sent back more than ``MAX_SANDBOX_OUTPUT_SIZE``."""


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
    def __init__(self) -> None:
        self._size = 0

    def __call__(self, output: str | bytes) -> None:
        self._size += len(output)
        if self._size > MAX_SANDBOX_OUTPUT_SIZE:
            raise SandboxOutputLimitError(
                "Sandbox output exceeds the limit of "
                f"{MAX_SANDBOX_OUTPUT_SIZE // 2**20} MiB"
            )


async def _count_body(response: httpx.Response) -> None:
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
    response.stream = _CountedBody(response.stream)


class _CountedBody(httpx.AsyncByteStream):
    def __init__(self, body: httpx.AsyncByteStream) -> None:
        self._body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        limit = _OutputLimit()
        async for chunk in self._body:
            limit(chunk)
            yield chunk

    async def aclose(self) -> None:
        await self._body.aclose()
