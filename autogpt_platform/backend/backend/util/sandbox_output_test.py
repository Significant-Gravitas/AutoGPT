"""How much of a sandbox's output reaches this process.

These drive the E2B SDK's own readers, since the limits rely on how they read."""

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from e2b import AsyncCommandHandle, AsyncSandbox, ConnectionConfig, SandboxException
from e2b.envd.process import process_pb
from packaging.version import Version
from protobuf import Oneof
from pyqwest import Headers, Request, Response

from backend.util.sandbox_output import (
    MAX_SANDBOX_MESSAGE_SIZE,
    MAX_SANDBOX_OUTPUT_SIZE,
    MAX_SANDBOX_RESPONSE_SIZE,
    OutputLimitedCodeSandbox,
    SandboxOutputLimitError,
    limit_sandbox_responses,
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


_STREAMED = "application/connect+json"


def _limited_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    body: AsyncIterator[bytes],
    content_type: str = _STREAMED,
    status: int = 200,
) -> AsyncSandbox:
    """A limited sandbox whose daemon answers every RPC with *body*."""

    class Pool:
        async def execute(self, request: Request) -> Response:
            headers = Headers({"content-type": content_type})
            return Response(status=status, headers=headers, content=body)

    monkeypatch.setattr(
        "backend.util.sandbox_output.get_pyqwest_transport", lambda *_, **__: Pool()
    )
    return limit_sandbox_responses(_plain_sandbox())


def _plain_sandbox() -> AsyncSandbox:
    return AsyncSandbox(
        sandbox_id="sb-1",
        envd_version=Version("0.2.0"),
        envd_access_token=None,
        sandbox_domain=None,
        connection_config=ConnectionConfig(api_key="k"),
    )


def _message(event: dict | None = None, flags: int = 0, **fields) -> bytes:
    """One message of a streamed answer, behind its flags and length."""
    data = json.dumps({"event": event, **fields} if event else fields).encode()
    return bytes([flags]) + len(data).to_bytes(4, "big") + data


_STARTED = _message({"start": {"pid": 1}})
_ENDED = _message({"end": {"exitCode": 0, "exited": True}}) + _message(flags=2)


@pytest.mark.asyncio
async def test_a_limited_sandbox_returns_output_within_the_limits(monkeypatch):
    answer = _STARTED + _message({"data": {"stdout": "aGVsbG8K"}}) + _ENDED

    async def body() -> AsyncIterator[bytes]:
        # In pieces that split the messages and their lengths.
        for start in range(0, len(answer), 3):
            yield answer[start : start + 3]

    result = await run_sandbox_command(_limited_sandbox(monkeypatch, body()), "echo")

    assert (result.stdout, result.exit_code) == ("hello\n", 0)


@pytest.mark.asyncio
async def test_a_limited_sandbox_stops_a_long_print_at_the_output_limit(monkeypatch):
    """In the daemon's 32 KiB pieces, the output limit is the one reached."""
    printed = _message({"data": {"stdout": "QUFB" * (32 * 1024 // 3)}})

    async def body() -> AsyncIterator[bytes]:
        yield _STARTED
        for _ in range(_PAST_THE_LIMIT * _MIB // (32 * 1024)):
            yield printed
        yield _ENDED

    with pytest.raises(SandboxOutputLimitError):
        await run_sandbox_command(_limited_sandbox(monkeypatch, body()), "yes")


@pytest.mark.asyncio
async def test_a_limited_sandbox_refuses_a_long_message_before_reading_it(
    monkeypatch,
):
    """The SDK would buffer the message whole and decode it in one go."""
    sent = _Sent()

    async def body() -> AsyncIterator[bytes]:
        yield _STARTED
        yield bytes([0]) + (MAX_SANDBOX_MESSAGE_SIZE + 1).to_bytes(4, "big")
        for _ in range(4):
            sent.size += _MIB
            yield b"A" * _MIB

    with pytest.raises(SandboxException, match="message over the limit"):
        await run_sandbox_command(_limited_sandbox(monkeypatch, body()), "true")

    assert sent.size == 0


@pytest.mark.asyncio
async def test_a_limited_sandbox_stops_messages_that_carry_no_output(monkeypatch):
    """A message without output never reaches the command's callbacks."""
    sent = _Sent()
    ignored = _message(x="A" * (_MIB // 2))

    async def body() -> AsyncIterator[bytes]:
        yield _STARTED
        for _ in range(MAX_SANDBOX_RESPONSE_SIZE // len(ignored) + 8):
            sent.size += len(ignored)
            yield ignored
        yield _ENDED

    with pytest.raises(SandboxException, match="response over the limit"):
        await run_sandbox_command(_limited_sandbox(monkeypatch, body()), "true")

    assert sent.size <= MAX_SANDBOX_RESPONSE_SIZE + _MIB


@pytest.mark.parametrize(
    "content_type, status",
    [("application/json", 200), ("application/json", 500), ("text/html", 502)],
)
@pytest.mark.asyncio
async def test_a_limited_sandbox_stops_an_answer_that_is_one_long_message(
    monkeypatch, content_type: str, status: int
):
    """A listing or an error, which the SDK reads whole before looking at it."""
    sent = _Sent()

    async def body() -> AsyncIterator[bytes]:
        yield b'{"entries": [], "x": "'
        for _ in range(MAX_SANDBOX_MESSAGE_SIZE // _MIB * 4 + 8):
            sent.size += _MIB // 4
            yield b"A" * (_MIB // 4)
        yield b'"}'

    sandbox = _limited_sandbox(monkeypatch, body(), content_type, status)

    with pytest.raises(SandboxException, match="response over the limit"):
        await sandbox.files.list("/etc/profile.d")

    assert sent.size <= MAX_SANDBOX_MESSAGE_SIZE + _MIB // 4


class _Body(httpx.AsyncByteStream):
    def __init__(self, chunks: AsyncIterator[bytes]):
        self._chunks = chunks
        self.closed = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._chunks

    async def aclose(self) -> None:
        self.closed = True


def _limited_sandbox_with_file_api(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    body: _Body,
    seen: list[httpx.Request] | None = None,
) -> AsyncSandbox:
    """A limited sandbox whose daemon answers every file API call with *body*."""

    def respond(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, stream=body)

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        "e2b.api.client_async.get_envd_transport", lambda *_, **__: transport
    )
    return limit_sandbox_responses(_plain_sandbox())


def _long_json(sent: _Sent, start: bytes, end: bytes):
    async def chunks() -> AsyncIterator[bytes]:
        yield start
        for _ in range(MAX_SANDBOX_MESSAGE_SIZE // _MIB * 4 + 8):
            sent.size += _MIB // 4
            yield b"A" * (_MIB // 4)
        yield end

    return chunks()


@pytest.mark.asyncio
async def test_a_limited_sandbox_leaves_a_file_to_its_reader(monkeypatch):
    """A file may be longer than any other answer: its reader has the limit."""
    seen: list[httpx.Request] = []

    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(MAX_SANDBOX_MESSAGE_SIZE // _MIB + 2):
            yield b"A" * _MIB

    sandbox = _limited_sandbox_with_file_api(monkeypatch, 200, _Body(chunks()), seen)

    data = await read_sandbox_file(sandbox, "/big", MAX_SANDBOX_MESSAGE_SIZE + 2 * _MIB)

    assert len(data) == MAX_SANDBOX_MESSAGE_SIZE + 2 * _MIB
    # Asked for uncompressed, so what is counted is what the SDK decodes.
    assert [r.headers.get("accept-encoding") for r in seen] == ["identity"]


@pytest.mark.parametrize("status", [404, 500])
@pytest.mark.asyncio
async def test_a_limited_sandbox_stops_a_long_error_from_the_file_api(
    monkeypatch, status: int
):
    """The SDK reads an error whole and decodes it before it raises."""
    sent = _Sent()
    body = _Body(_long_json(sent, b'{"message": "', b'"}'))
    sandbox = _limited_sandbox_with_file_api(monkeypatch, status, body)

    with pytest.raises(SandboxOutputLimitError):
        await read_sandbox_file(sandbox, "/etc/profile", 16)

    assert sent.size <= MAX_SANDBOX_MESSAGE_SIZE + _MIB // 4
    # The SDK leaves a response whose error it could not read to its reader.
    assert body.closed


@pytest.mark.asyncio
async def test_a_limited_sandbox_stops_a_long_answer_to_a_write(monkeypatch):
    """The SDK reads the answer whole and decodes it as the files written."""
    sent = _Sent()
    body = _Body(_long_json(sent, b'[{"name": "f", "type": "file", "x": "', b'"}]'))
    sandbox = _limited_sandbox_with_file_api(monkeypatch, 200, body)

    with pytest.raises(SandboxOutputLimitError):
        await sandbox.files.write("/f", "content")

    assert sent.size <= MAX_SANDBOX_MESSAGE_SIZE + _MIB // 4


@pytest.mark.asyncio
async def test_a_limited_sandbox_stops_a_long_answer_to_a_health_check(monkeypatch):
    """Not a file, though it is also a GET answered with 200."""
    sent = _Sent()
    sandbox = _limited_sandbox_with_file_api(
        monkeypatch, 200, _Body(_long_json(sent, b"", b""))
    )

    with pytest.raises(SandboxOutputLimitError):
        await sandbox.is_running()

    assert sent.size <= MAX_SANDBOX_MESSAGE_SIZE + _MIB // 4


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
