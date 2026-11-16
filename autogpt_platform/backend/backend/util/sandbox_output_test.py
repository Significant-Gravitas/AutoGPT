"""How much of a sandbox's output reaches this process.

These drive the E2B SDK's own readers, since the limits rely on how they read."""

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from e2b import AsyncCommandHandle, ConnectionConfig
from e2b.envd.process import process_pb
from packaging.version import Version
from protobuf import Oneof

from backend.util.sandbox_output import (
    MAX_SANDBOX_OUTPUT_SIZE,
    OutputLimitedCodeSandbox,
    SandboxOutputLimitError,
    read_sandbox_file,
    run_sandbox_command,
)

_MIB = 2**20
# Streams stop a little past the limit, so a missing limit fails the test fast.
_PAST_THE_LIMIT = MAX_SANDBOX_OUTPUT_SIZE // _MIB + 4


class _Sent:
    size = 0


async def _run_code(
    body: AsyncIterator[bytes],
    headers: dict[str, str] | None = None,
    seen: list[httpx.Request] | None = None,
):
    sandbox = OutputLimitedCodeSandbox(
        sandbox_id="sb-1",
        envd_version=Version("0.2.0"),
        envd_access_token=None,
        sandbox_domain=None,
        connection_config=ConnectionConfig(api_key="k"),
    )

    def respond(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, content=body, headers=headers)

    transport = httpx.MockTransport(respond)
    with patch(
        "e2b_code_interpreter.code_interpreter_async.get_transport",
        return_value=transport,
    ):
        return await sandbox.run_code("print('hi')")


def _stdout(text: str) -> bytes:
    return json.dumps({"type": "stdout", "text": text, "timestamp": 0}).encode() + b"\n"


@pytest.mark.asyncio
async def test_run_code_returns_output_within_the_limit():
    async def body() -> AsyncIterator[bytes]:
        yield _stdout("hello ")
        yield _stdout("world\n")

    execution = await _run_code(body())

    assert execution.logs.stdout == ["hello ", "world\n"]


@pytest.mark.asyncio
async def test_run_code_gives_up_on_a_program_that_keeps_printing():
    sent = _Sent()

    async def body() -> AsyncIterator[bytes]:
        for _ in range(_PAST_THE_LIMIT):
            sent.size += _MIB
            yield _stdout("A" * _MIB)

    with pytest.raises(SandboxOutputLimitError):
        await _run_code(body())

    assert sent.size <= MAX_SANDBOX_OUTPUT_SIZE + 2 * _MIB


@pytest.mark.asyncio
async def test_run_code_asks_for_an_uncompressed_body():
    async def body() -> AsyncIterator[bytes]:
        yield _stdout("hi\n")

    seen: list[httpx.Request] = []
    await _run_code(body(), seen=seen)

    assert [r.headers.get("accept-encoding") for r in seen] == ["identity"]


@pytest.mark.asyncio
async def test_run_code_refuses_a_body_it_cannot_meter_while_reading():
    """A compressed body decodes above the counted stream, so it is refused."""
    sent = _Sent()

    async def body() -> AsyncIterator[bytes]:
        for _ in range(_PAST_THE_LIMIT):
            sent.size += _MIB
            yield b"A" * _MIB

    with pytest.raises(SandboxOutputLimitError, match="gzip"):
        await _run_code(body(), headers={"Content-Encoding": "gzip"})

    assert sent.size == 0


@pytest.mark.asyncio
async def test_run_code_gives_up_on_a_message_that_never_ends():
    sent = _Sent()

    async def body() -> AsyncIterator[bytes]:
        yield b'{"type": "stdout", "timestamp": 0, "text": "'
        for _ in range(_PAST_THE_LIMIT):
            sent.size += _MIB
            yield b"A" * _MIB

    with pytest.raises(SandboxOutputLimitError):
        await _run_code(body())

    assert sent.size <= MAX_SANDBOX_OUTPUT_SIZE + _MIB


def _sandbox_running(events: AsyncIterator[process_pb.StartResponse]) -> MagicMock:
    async def run(command: str, *, on_stdout, on_stderr, **kwargs):
        handle = AsyncCommandHandle(
            pid=1,
            handle_kill=AsyncMock(),
            events=events,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )
        return await handle.wait()

    sandbox = MagicMock()
    sandbox.commands.run = run
    return sandbox


def _event(field: str, value) -> process_pb.StartResponse:
    return process_pb.StartResponse(
        event=process_pb.ProcessEvent(event=Oneof(field, value))
    )


def _printed(stream: str, data: bytes) -> process_pb.StartResponse:
    return _event("data", process_pb.ProcessEvent.DataEvent(output=Oneof(stream, data)))


@pytest.mark.asyncio
async def test_run_sandbox_command_returns_output_within_the_limit():
    async def events() -> AsyncIterator[process_pb.StartResponse]:
        yield _printed("stdout", b"out")
        yield _printed("stderr", b"err")
        yield _event("end", process_pb.ProcessEvent.EndEvent(exit_code=0, exited=True))

    result = await run_sandbox_command(_sandbox_running(events()), "echo out")

    assert (result.stdout, result.stderr, result.exit_code) == ("out", "err", 0)


@pytest.mark.asyncio
async def test_run_sandbox_command_gives_up_on_a_command_that_keeps_printing():
    sent = _Sent()

    async def events() -> AsyncIterator[process_pb.StartResponse]:
        for _ in range(_PAST_THE_LIMIT):
            # Both streams draw on the one limit.
            for stream in ("stdout", "stderr"):
                sent.size += _MIB
                yield _printed(stream, b"A" * _MIB)
        yield _event("end", process_pb.ProcessEvent.EndEvent(exit_code=0, exited=True))

    with pytest.raises(SandboxOutputLimitError):
        await run_sandbox_command(_sandbox_running(events()), "yes")

    assert sent.size == MAX_SANDBOX_OUTPUT_SIZE + _MIB


class _Stream:
    def __init__(self, chunks: AsyncIterator[bytes]):
        self._chunks = chunks
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    def __aiter__(self):
        return self._chunks


def _sandbox_with_file(chunks: AsyncIterator[bytes]) -> tuple[MagicMock, _Stream]:
    stream = _Stream(chunks)
    sandbox = MagicMock()
    sandbox.files.read = AsyncMock(return_value=stream)
    return sandbox, stream


@pytest.mark.asyncio
async def test_read_sandbox_file_stops_one_byte_past_the_limit():
    sent = _Sent()

    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(100):
            sent.size += 4
            yield b"abcd"

    sandbox, stream = _sandbox_with_file(chunks())

    data = await read_sandbox_file(sandbox, "/big", 10)

    assert data == b"abcdabcdabc"
    assert sent.size == 12
    assert stream.closed
    sandbox.files.read.assert_awaited_once_with("/big", format="stream")


@pytest.mark.asyncio
async def test_read_sandbox_file_returns_a_file_within_the_limit_whole():
    async def chunks() -> AsyncIterator[bytes]:
        yield b"abcd"
        yield b"efgh"

    sandbox, _ = _sandbox_with_file(chunks())

    assert await read_sandbox_file(sandbox, "/small", 8) == b"abcdefgh"
