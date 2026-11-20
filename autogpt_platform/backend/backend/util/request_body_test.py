"""A request's body goes out whole on every send: again after a throttle
status, and on to the next hop of a redirect (#16954, #17130)."""

import asyncio
import contextlib
import io
import os
import tempfile
import threading
import time
import tracemalloc
from io import BytesIO
from typing import Any, AsyncIterator, BinaryIO, Callable, ContextManager, Iterator
from urllib.parse import ParseResult, urlencode

import pytest
from aiohttp import BodyPartReader, ClientSession, ClientTimeout, FormData, web
from aiohttp.payload import BufferedReaderPayload
from multidict import MultiDict

from backend.util import request as request_module
from backend.util import request_body
from backend.util.request import Requests
from backend.util.request_body import RequestBodyTimeoutError, RequestBodyTooLargeError
from backend.util.request_test_helpers import (
    ENDS_WITHIN_SECONDS,
    RETRYING_CLIENTS,
    local_requests,
    scripted,
    serve,
)

# Under one 4 KiB page: a pipe can be that small (pipe-user-pages-soft), and
# the pipe case writes the whole payload before anything reads it.
_PAYLOAD = b"#1=CARTESIAN_POINT('',(0.,0.,0.));\n" * 100
_STEP_HEADER = b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n"
# One page: the least a Linux pipe holds at the soft limit; macOS starts at 16 KiB.
_PIPE_CAPACITY = 4096


def _bytes(payload: bytes) -> ContextManager[bytes]:
    return contextlib.nullcontext(payload)


def _memoryview(payload: bytes) -> ContextManager[memoryview]:
    return contextlib.nullcontext(memoryview(payload))


def _bytesio(payload: bytes) -> BinaryIO:
    return BytesIO(payload)


def _stringio(payload: bytes) -> io.StringIO:
    return io.StringIO(payload.decode())


def _temp_file(payload: bytes) -> BinaryIO:
    """A real file on disk: aiohttp closes it once it has sent it."""
    handle = tempfile.TemporaryFile()
    handle.write(payload)
    handle.seek(0)
    return handle


def _named_file(payload: bytes) -> BinaryIO:
    """A file on disk named ``drawing.png``, from which aiohttp takes a part's
    filename and a Content-Type."""
    return _opened_after_writing(payload, "drawing.png", "rb")


def _text_file(payload: bytes) -> io.TextIOWrapper:
    """A file on disk named ``notes.txt``, opened as text."""
    return _opened_after_writing(payload, "notes.txt", "r")


def _opened_after_writing(payload: bytes, name: str, mode: str) -> Any:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, name)
        with open(path, "wb") as handle:
            handle.write(payload)
        # Still readable once its directory is gone.
        return open(path, mode, encoding=None if "b" in mode else "utf-8")


def _bytesio_past_what_was_read(payload: bytes) -> BinaryIO:
    """A handle the caller already read a header from: the body is the rest."""
    handle = BytesIO(_STEP_HEADER + payload)
    handle.seek(len(_STEP_HEADER))
    return handle


def _pipe(payload: bytes) -> BinaryIO:
    """A stream that can't seek, as a pipe or a socket can't."""
    # The pipe holds all of the payload, so writing it can't block waiting
    # for a reader.
    assert len(payload) <= _PIPE_CAPACITY
    read_end, write_end = os.pipe()
    try:
        os.write(write_end, payload)
    finally:
        os.close(write_end)
    return open(read_end, "rb")


async def _lines(payload: bytes) -> AsyncIterator[bytes]:
    """*payload* as an async iterable, a line at a time."""
    for line in payload.splitlines(keepends=True):
        yield line


# The forms a caller can pass as ``data``, each with a text field before the
# file: aiohttp sends all of them the same way.
_FORMS = {
    "form-dict": lambda file: {"note": "kept", "file": file},
    "form-pairs": lambda file: [("note", "kept"), ("file", file)],
    "form-multidict": lambda file: MultiDict([("note", "kept"), ("file", file)]),
    "form-bare-file": lambda file: [("note", "kept"), file],
}


def _sent_as(send: str, body: Any) -> dict:
    """*body* as an upload, as the whole ``data``, or as a form's file."""
    if send == "files":
        return {"files": [("file", ("part.step", body, "model/step"))]}
    if send == "data":
        return {"data": body}
    return {"data": _FORMS[send](body)}


async def _received(request: web.Request) -> Any:
    """What a request carried: a multipart form's fields by name, or else its
    whole body."""
    if not request.content_type.startswith("multipart/"):
        return await request.read()
    fields = {}
    reader = await request.multipart()
    while (part := await reader.next()) is not None:
        assert isinstance(part, BodyPartReader)
        fields[part.name] = await part.read()
    return fields


def _expected(send: str, payload: bytes) -> Any:
    """What a server receives when ``_sent_as(send, ...)`` sends *payload*."""
    if send == "data":
        return payload
    if send == "files":
        return {"file": payload}
    return {"note": b"kept", "file": payload}


def _receiving(*statuses: int) -> tuple[web.Application, list[Any]]:
    """An app taking POSTs to ``/``, answering *statuses* in turn (the last
    for good), and what each request carried."""
    received: list[Any] = []

    async def receive(request: web.Request) -> web.Response:
        received.append(await _received(request))
        return web.Response(status=statuses[min(len(received), len(statuses)) - 1])

    app = web.Application()
    app.router.add_post("/", receive)
    return app, received


@pytest.mark.asyncio
@RETRYING_CLIENTS
@pytest.mark.parametrize(
    "open_body",
    [_bytes, _memoryview, _bytesio, _bytesio_past_what_was_read, _temp_file, _pipe],
    ids=[
        "bytes",
        "memoryview",
        "bytesio",
        "bytesio-past-what-was-read",
        "temp-file",
        "stream-that-cannot-seek",
    ],
)
@pytest.mark.parametrize("send", ["files", "data", "form-dict"])
async def test_a_body_retried_after_a_throttle_status_is_sent_whole_each_time(
    send: str, open_body: Callable[[bytes], ContextManager[Any]], client_kwargs: dict
):
    app, received = _receiving(429, 200)
    runner, base = await serve(app)

    try:
        with open_body(_PAYLOAD) as body:
            response = await asyncio.wait_for(
                local_requests(retry_max_wait=0, **client_kwargs).post(
                    base, **_sent_as(send, body)
                ),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        await runner.cleanup()

    assert response.status == 200
    assert received == [_expected(send, _PAYLOAD)] * 2


@pytest.mark.asyncio
@RETRYING_CLIENTS
async def test_an_async_iterable_body_retried_after_a_throttle_status_is_sent_whole(
    client_kwargs: dict,
):
    app, received = _receiving(429, 200)
    runner, base = await serve(app)

    try:
        response = await asyncio.wait_for(
            local_requests(retry_max_wait=0, **client_kwargs).post(
                base, data=_lines(_PAYLOAD)
            ),
            ENDS_WITHIN_SECONDS,
        )
    finally:
        await runner.cleanup()

    assert response.status == 200
    assert received == [_PAYLOAD, _PAYLOAD]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "open_body",
    [_bytesio, _temp_file, _pipe],
    ids=["bytesio", "temp-file", "stream-that-cannot-seek"],
)
@pytest.mark.parametrize("send", ["files", "data", "form-dict"])
async def test_a_body_redirected_with_a_307_reaches_the_next_hop_whole(
    send: str, open_body: Callable[[bytes], ContextManager[Any]]
):
    """A 307 repeats the request, body and all, so the hop must not find a
    stream where the first send left it, at its end, or closed. A one-attempt
    client follows redirects too, so it keeps the body for the hop."""
    received: dict[str, Any] = {}

    async def moved(request: web.Request) -> web.Response:
        received[request.path] = await _received(request)
        raise web.HTTPTemporaryRedirect("/final")

    async def final(request: web.Request) -> web.Response:
        received[request.path] = await _received(request)
        return web.Response(text="stored")

    app = web.Application()
    app.router.add_post("/", moved)
    app.router.add_post("/final", final)
    runner, base = await serve(app)

    try:
        with open_body(_PAYLOAD) as body:
            response = await asyncio.wait_for(
                local_requests().post(base, **_sent_as(send, body)),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        await runner.cleanup()

    assert response.status == 200
    sent = _expected(send, _PAYLOAD)
    assert received == {"/": sent, "/final": sent}


# --- What a body carries on a send again is what it would carry once ---


async def _carried(request: web.Request) -> Any:
    """All a request's body carried: each part of a multipart form, with its
    filename and Content-Type, or else the body and its Content-Type."""
    if not request.content_type.startswith("multipart/"):
        return request.headers.get("Content-Type"), await request.read()
    parts = []
    reader = await request.multipart()
    while (part := await reader.next()) is not None:
        assert isinstance(part, BodyPartReader)
        content_type = part.headers.get("Content-Type")
        parts.append((part.name, part.filename, content_type, await part.read()))
    return parts


async def _carried_twice_and_once(
    how: str, sent: Callable[[], ContextManager[dict]]
) -> tuple[list[Any], Any]:
    """What a body carried on both sends of a request that sends it twice,
    ``throttled`` (a 429, then a 200) or ``redirected`` (a 307), and on a
    request that sends it once, which aiohttp streams as it is."""
    carried: dict[str, list[Any]] = {"/once": [], "/": [], "/again": []}

    async def record(request: web.Request) -> web.Response:
        carried[request.path].append(await _carried(request))
        if request.path == "/" and how == "redirected":
            raise web.HTTPTemporaryRedirect("/again")
        if request.path == "/" and len(carried["/"]) == 1:
            return web.Response(status=429)
        return web.Response(text="stored")

    app = web.Application()
    app.router.add_post("/{path:.*}", record)
    runner, base = await serve(app)
    twice = (
        local_requests(retry_max_attempts=2, retry_max_wait=0)
        if how == "throttled"
        else local_requests()
    )

    try:
        with sent() as body:
            await asyncio.wait_for(
                local_requests().post(f"{base}/once", allow_redirects=False, **body),
                ENDS_WITHIN_SECONDS,
            )
        with sent() as body:
            await asyncio.wait_for(twice.post(f"{base}/", **body), ENDS_WITHIN_SECONDS)
    finally:
        await runner.cleanup()

    return carried["/"] + carried["/again"], carried["/once"][0]


@contextlib.contextmanager
def _form_of(shape: str, stream: Any) -> Iterator[dict]:
    with stream:
        yield {"data": _FORMS[shape](stream)}


_HOW = pytest.mark.parametrize("how", ["throttled", "redirected"])


@pytest.mark.asyncio
@_HOW
@pytest.mark.parametrize(
    "open_stream",
    [
        _bytesio,
        _bytesio_past_what_was_read,
        _stringio,
        _named_file,
        _text_file,
        _pipe,
    ],
    ids=[
        "bytesio",
        "bytesio-past-what-was-read",
        "stringio",
        "named-file",
        "text-file",
        "stream-that-cannot-seek",
    ],
)
@pytest.mark.parametrize("shape", list(_FORMS))
async def test_a_stream_in_a_form_goes_out_on_every_send_as_it_would_once(
    shape: str, open_stream: Callable[[bytes], Any], how: str
):
    """A ``BytesIO`` or ``StringIO`` in a form is rewound for each send, and
    any other file is read into memory under its own name: every send carries
    the same parts, filenames and Content-Types as a request that sends the
    form once."""
    sends, once = await _carried_twice_and_once(
        how, lambda: _form_of(shape, open_stream(_PAYLOAD))
    )

    assert sends == [once, once]
    assert ("note", None, "text/plain; charset=utf-8", b"kept") in once
    assert _PAYLOAD in [content for *_, content in once]


@pytest.mark.asyncio
@_HOW
@pytest.mark.parametrize(
    "open_file, filename, content_type",
    [
        (_named_file, "drawing.png", "image/png"),
        (_text_file, "notes.txt", "text/plain; charset=utf-8"),
    ],
    ids=["named-file", "text-file"],
)
@pytest.mark.parametrize("send", ["data", "files", "form-dict"])
async def test_a_file_read_into_memory_keeps_its_name_and_content_type(
    send: str,
    open_file: Callable[[bytes], Any],
    filename: str,
    content_type: str,
    how: str,
):
    """aiohttp takes a part's filename, and a body's or a part's Content-Type,
    from a file's name; the file held in memory in its place keeps it."""

    @contextlib.contextmanager
    def sent() -> Iterator[dict]:
        with open_file(_PAYLOAD) as file:
            yield _sent_as(send, file)

    sends, once = await _carried_twice_and_once(how, sent)

    assert sends == [once, once]
    if send == "data":
        assert once == (content_type, _PAYLOAD)
    elif send == "files":
        assert once == [("file", "part.step", "model/step", _PAYLOAD)]
    else:
        assert ("file", filename, content_type, _PAYLOAD) in once


def _form_data(value: Any, **field: Any) -> FormData:
    form = FormData()
    form.add_field("note", "kept")
    form.add_field("file", value, **field)
    return form


def _note_and_file(filename: str, content_type: str) -> list[tuple]:
    """The parts of a ``_form_data`` form: its text field, then its file."""
    return [
        ("note", None, "text/plain; charset=utf-8", b"kept"),
        ("file", filename, content_type, _PAYLOAD),
    ]


@pytest.mark.asyncio
@_HOW
@pytest.mark.parametrize(
    "make_form, carried",
    [
        (
            lambda: _form_data(BytesIO(_PAYLOAD)),
            _note_and_file("file", "application/octet-stream"),
        ),
        (
            lambda: _form_data(io.StringIO(_PAYLOAD.decode())),
            _note_and_file("file", "text/plain; charset=utf-8"),
        ),
        (
            lambda: _form_data(_PAYLOAD, filename="part.step"),
            _note_and_file("part.step", "application/octet-stream"),
        ),
        (
            lambda: FormData({"note": "kept", "file": _PAYLOAD.decode()}),
            (
                "application/x-www-form-urlencoded",
                urlencode({"note": "kept", "file": _PAYLOAD.decode()}).encode(),
            ),
        ),
    ],
    ids=["bytesio", "stringio", "bytes", "text-fields"],
)
async def test_a_formdata_aiohttp_can_send_again_goes_out_whole_on_every_send(
    make_form: Callable[[], FormData], carried: Any, how: str
):
    """aiohttp encodes a ``FormData`` once and sends its parts again, all of
    a ``BytesIO`` or ``StringIO`` each time, so it goes as it is."""
    sends, once = await _carried_twice_and_once(
        how, lambda: contextlib.nullcontext({"data": make_form()})
    )

    assert once == carried
    assert sends == [carried, carried]


@contextlib.contextmanager
def _part_aiohttp_uses_up(part: str) -> Iterator[dict]:
    """A ``_form_data`` file part that aiohttp uses up in sending it: a file,
    which it closes, or an async iterable, which runs once."""
    if part == "async-iterable":
        yield {"value": _lines(_PAYLOAD), "filename": "file"}
        return
    with _temp_file(_PAYLOAD) as file:
        yield {"value": file}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_kwargs, send_kwargs",
    [
        ({"retry_max_attempts": 3}, {"allow_redirects": False}),
        ({"raise_for_status": False}, {"allow_redirects": False}),
        ({}, {}),
    ],
    ids=["retried-raising", "retried-status-checked", "redirects-followed"],
)
@pytest.mark.parametrize("part", ["file", "async-iterable"])
async def test_a_formdata_holding_a_part_aiohttp_uses_up_is_refused_when_it_might_be(
    part: str, client_kwargs: dict, send_kwargs: dict
):
    """aiohttp sends a ``FormData``'s parts again, but by then it has closed a
    file among them or run an async iterable, so the retry or the redirected
    hop would fail or send an empty part. A request that might send the form
    again refuses it before sending anything; one that sends once sends it."""
    app, received = _receiving(200)
    runner, base = await serve(app)

    try:
        with _part_aiohttp_uses_up(part) as field:
            with pytest.raises(TypeError, match="FormData body can't be sent again"):
                await asyncio.wait_for(
                    local_requests(retry_max_wait=0, **client_kwargs).post(
                        base, data=_form_data(**field), **send_kwargs
                    ),
                    ENDS_WITHIN_SECONDS,
                )
        assert received == []

        with _part_aiohttp_uses_up(part) as field:
            response = await asyncio.wait_for(
                local_requests().post(
                    base, allow_redirects=False, data=_form_data(**field)
                ),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        await runner.cleanup()

    assert response.status == 200
    assert received == [{"note": b"kept", "file": _PAYLOAD}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_value, kind",
    [
        (lambda: _lines(_PAYLOAD), "async_generator"),
        (lambda: BufferedReaderPayload(BytesIO(_PAYLOAD)), "BufferedReaderPayload"),
    ],
    ids=["async-iterable", "aiohttp-payload"],
)
async def test_a_form_value_with_no_way_to_be_sent_again_is_refused_when_it_might_be(
    make_value: Callable[[], Any], kind: str
):
    app, served = scripted(200)
    runner, base = await serve(app)

    try:
        with pytest.raises(TypeError, match=f"{kind} form value can't be sent again"):
            await asyncio.wait_for(
                local_requests().post(
                    base, data={"note": "kept", "file": make_value()}
                ),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        await runner.cleanup()

    assert served == []


@pytest.mark.asyncio
async def test_a_body_of_another_type_is_refused_before_anything_is_sent():
    """Neither bytes, text, a form, a file nor an async iterable: nothing says
    how to send it again."""
    app, served = scripted(200)
    runner, base = await serve(app)

    try:
        with pytest.raises(TypeError, match="list_iterator body can't be sent again"):
            await asyncio.wait_for(
                local_requests().post(base, data=iter([_PAYLOAD])),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        await runner.cleanup()

    assert served == []


# --- A body held in memory is bounded in size and in time ---


@contextlib.contextmanager
def _sent_once(send: str, payload: bytes) -> Iterator[dict]:
    """A body that can't be read twice: a pipe, as an upload, as ``data`` or
    in a form, or an async iterable."""
    if send == "async-iterable":
        yield {"data": _lines(payload)}
        return
    with _pipe(payload) as stream:
        yield _sent_as(send, stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("send", ["files", "data", "form-dict", "async-iterable"])
async def test_a_body_that_cannot_be_read_twice_is_refused_past_the_limit(
    send: str, monkeypatch: pytest.MonkeyPatch
):
    """It has to be held in memory to be sent again, so past the limit the
    request fails before anything is sent."""
    monkeypatch.setattr(request_body, "MAX_BUFFERED_BODY_BYTES", 1024)
    app, served = scripted(200)
    runner, base = await serve(app)

    try:
        with _sent_once(send, _PAYLOAD) as body:
            with pytest.raises(RequestBodyTooLargeError):
                await asyncio.wait_for(
                    local_requests().post(base, **body), ENDS_WITHIN_SECONDS
                )
    finally:
        await runner.cleanup()

    assert served == []


@pytest.mark.asyncio
async def test_the_limit_counts_every_part_held_in_memory(
    monkeypatch: pytest.MonkeyPatch,
):
    """Two files each under the limit, but over it together, are refused; one
    of them alone goes."""
    monkeypatch.setattr(request_body, "MAX_BUFFERED_BODY_BYTES", 1024)
    half = _PAYLOAD[:600]
    app, received = _receiving(200)
    runner, base = await serve(app)

    try:
        with _pipe(half) as first, _pipe(half) as second:
            with pytest.raises(RequestBodyTooLargeError):
                await asyncio.wait_for(
                    local_requests().post(
                        base, data={"first": first, "second": second}
                    ),
                    ENDS_WITHIN_SECONDS,
                )
        assert received == []

        with _pipe(half) as first:
            response = await asyncio.wait_for(
                local_requests().post(base, data={"first": first}),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        await runner.cleanup()

    assert response.status == 200
    assert received == [{"first": half}]


async def _stalls_after_a_chunk(released: list[bool]) -> AsyncIterator[bytes]:
    """An async iterable body that yields a chunk and then waits for good,
    noting when it is let go."""
    try:
        yield _PAYLOAD[:64]
        await asyncio.Event().wait()
    finally:
        released.append(True)


_REQUEST_TIMEOUTS = pytest.mark.parametrize(
    "timeout_kwargs",
    [{"timeout": ClientTimeout(total=0.05)}, {"timeout": 0.05}, {}],
    ids=["client-timeout", "seconds", "the-sessions-own"],
)


@pytest.mark.asyncio
@_REQUEST_TIMEOUTS
async def test_an_async_iterable_body_that_stalls_fails_within_the_requests_timeout(
    timeout_kwargs: dict, monkeypatch: pytest.MonkeyPatch
):
    """The body is read before the first send, outside aiohttp's timeout, so
    the request's own timeout bounds the read: the request fails in that
    time, the stalled iterable is let go, and nothing is sent."""
    monkeypatch.setattr(request_module, "DEFAULT_TIMEOUT", ClientTimeout(total=0.05))
    released: list[bool] = []
    app, served = scripted(200)
    runner, base = await serve(app)

    try:
        with pytest.raises(RequestBodyTimeoutError):
            await asyncio.wait_for(
                local_requests().post(
                    base, data=_stalls_after_a_chunk(released), **timeout_kwargs
                ),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        await runner.cleanup()

    assert released == [True]
    assert served == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timeout",
    [None, 0, ClientTimeout(total=None), ClientTimeout(total=0)],
    ids=["none", "zero", "client-timeout-none", "client-timeout-zero"],
)
async def test_a_request_aiohttp_gives_no_time_limit_reads_its_body_without_one(
    timeout: Any,
):
    """aiohttp reads ``None`` and 0 as no limit on a send, so reading the body
    has none either."""
    app, received = _receiving(200)
    runner, base = await serve(app)

    try:
        with _pipe(_PAYLOAD) as stream:
            response = await asyncio.wait_for(
                local_requests().post(base, data=stream, timeout=timeout),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        await runner.cleanup()

    assert response.status == 200
    assert received == [_PAYLOAD]


@pytest.mark.asyncio
async def test_a_request_without_a_timeout_reads_its_body_within_the_sessions():
    async with ClientSession() as session:
        assert session.timeout is request_module.DEFAULT_TIMEOUT


class _StalledStream(io.RawIOBase):
    """A file whose read waits, as a pipe's does while its writer writes
    nothing, until the test lets it go. After ``ENDS_WITHIN_SECONDS`` it ends
    by itself, so a read made on the event loop fails the test instead of
    blocking it for good."""

    def __init__(self):
        self.let_go = threading.Event()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        self.let_go.wait(ENDS_WITHIN_SECONDS)
        return 0


@pytest.mark.asyncio
async def test_a_file_that_stalls_fails_within_the_requests_timeout():
    """A file whose read, in a worker thread, would wait for good."""
    stream = _StalledStream()
    app, served = scripted(200)
    runner, base = await serve(app)

    try:
        with pytest.raises(RequestBodyTimeoutError):
            await asyncio.wait_for(
                local_requests().post(
                    base, data=stream, timeout=ClientTimeout(total=0.05)
                ),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        # Lets the worker thread's read return.
        stream.let_go.set()
        await runner.cleanup()

    assert served == []


@pytest.mark.asyncio
async def test_stalled_file_reads_leave_the_default_executor_free():
    """A read that stalls can't be interrupted, so it must not hold one of the
    event loop's few default-executor workers: DNS lookups and file I/O wait
    behind them."""
    app, served = scripted(200)
    runner, base = await serve(app)
    streams = [_StalledStream() for _ in range(48)]

    try:
        outcomes = await asyncio.gather(
            *(
                local_requests().post(
                    base, data=stream, timeout=ClientTimeout(total=0.05)
                )
                for stream in streams
            ),
            return_exceptions=True,
        )
        assert all(isinstance(outcome, RequestBodyTimeoutError) for outcome in outcomes)
        assert await asyncio.wait_for(asyncio.to_thread(int, "7"), 2) == 7
    finally:
        for stream in streams:
            stream.let_go.set()
        await runner.cleanup()

    assert served == []


class _EndlessStream(io.RawIOBase):
    """A file that never ends, counting its reads."""

    def __init__(self):
        self.reads = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        self.reads += 1
        time.sleep(0.001)
        return len(buffer)


@pytest.mark.asyncio
async def test_a_read_that_timed_out_does_not_read_on():
    """Nothing is waiting for what an abandoned read gathers, so it stops at
    its next chunk instead of filling memory up to the limit."""
    stream = _EndlessStream()
    app, served = scripted(200)
    runner, base = await serve(app)

    try:
        with pytest.raises(RequestBodyTimeoutError):
            await asyncio.wait_for(
                local_requests().post(
                    base, data=stream, timeout=ClientTimeout(total=0.1)
                ),
                ENDS_WITHIN_SECONDS,
            )
        reads_at_timeout = stream.reads
        await asyncio.sleep(0.3)
    finally:
        await runner.cleanup()

    assert stream.reads - reads_at_timeout <= 1
    assert served == []


# --- How a body is read into memory ---


_HELD_BYTES = 8 * 1024 * 1024


async def _chunks_of(payload: bytes) -> AsyncIterator[bytes]:
    for start in range(0, len(payload), 64 * 1024):
        yield payload[start : start + 64 * 1024]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["file", "async-iterable"])
async def test_a_body_held_in_memory_is_held_once(source: str):
    """Each chunk goes into one stream as it is read, so holding a body takes
    about its own size at the peak, not twice that for a copy."""
    payload = os.urandom(_HELD_BYTES)
    with tempfile.TemporaryFile() as file:
        file.write(payload)
        file.seek(0)
        chunks = _chunks_of(payload)
        tracemalloc.start()
        try:
            if source == "file":
                held, size = request_body._read_up_to(file, _HELD_BYTES)
            else:
                held, size = await request_body._gather_up_to(chunks, _HELD_BYTES)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    assert size == _HELD_BYTES
    assert held.read() == payload
    assert peak < 1.5 * _HELD_BYTES


class _OneWayStream(io.RawIOBase):
    """A stream that can't seek, like a pipe or a socket. It counts its reads,
    and notes one made on the event loop's own thread, where a blocking read
    would hold up every other request."""

    def __init__(self, payload: bytes):
        self._rest = payload
        self._loop_thread = threading.get_ident()
        self.reads = 0
        self.read_on_the_loop = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        self.reads += 1
        if threading.get_ident() == self._loop_thread:
            self.read_on_the_loop = True
        size = min(len(buffer), len(self._rest))
        buffer[:size] = self._rest[:size]
        self._rest = self._rest[size:]
        return size


@pytest.mark.asyncio
@pytest.mark.parametrize("send", ["files", "data", "form-dict"])
async def test_a_stream_that_cannot_seek_is_read_off_the_event_loop(send: str):
    stream = _OneWayStream(_PAYLOAD)
    app, received = _receiving(200)
    runner, base = await serve(app)

    try:
        response = await asyncio.wait_for(
            local_requests().post(base, **_sent_as(send, stream)),
            ENDS_WITHIN_SECONDS,
        )
    finally:
        await runner.cleanup()

    assert not stream.read_on_the_loop
    assert response.status == 200
    assert received == [_expected(send, _PAYLOAD)]


@pytest.mark.asyncio
@pytest.mark.parametrize("send", ["files", "data", "form-dict"])
async def test_a_blocked_url_is_refused_before_the_body_is_read(send: str):
    stream = _OneWayStream(_PAYLOAD)

    with pytest.raises(ValueError, match="blocked or private IP"):
        await asyncio.wait_for(
            Requests().post("http://127.0.0.1:1/", **_sent_as(send, stream)),
            ENDS_WITHIN_SECONDS,
        )

    assert stream.reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("send", ["files", "data", "form-dict"])
async def test_a_url_the_callers_validator_refuses_is_refused_before_the_body_is_read(
    send: str,
):
    stream = _OneWayStream(_PAYLOAD)

    def refuse(url: ParseResult) -> ParseResult:
        raise ValueError(f"Not an allowed host: {url.hostname}")

    with pytest.raises(ValueError, match="Not an allowed host: 127.0.0.1"):
        await asyncio.wait_for(
            local_requests(extra_url_validator=refuse).post(
                "http://127.0.0.1:1/", **_sent_as(send, stream)
            ),
            ENDS_WITHIN_SECONDS,
        )

    assert stream.reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("send", ["files", "data", "form-dict"])
async def test_a_request_that_sends_once_streams_its_body_unbuffered(
    send: str, monkeypatch: pytest.MonkeyPatch
):
    """One attempt and no redirects: nothing is sent twice, so nothing is held
    in memory, and a body past the buffer limit still goes out whole."""
    monkeypatch.setattr(request_body, "MAX_BUFFERED_BODY_BYTES", 1024)
    app, received = _receiving(200)
    runner, base = await serve(app)

    try:
        with _pipe(_PAYLOAD) as stream:
            response = await asyncio.wait_for(
                local_requests().post(
                    base, allow_redirects=False, **_sent_as(send, stream)
                ),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        await runner.cleanup()

    assert response.status == 200
    assert received == [_expected(send, _PAYLOAD)]


@pytest.mark.asyncio
@RETRYING_CLIENTS
async def test_a_stream_reader_body_retried_after_a_throttle_status_is_sent_whole(
    client_kwargs: dict,
):
    """A download's ``StreamReader``, sent on as the body of another request."""
    app, received = _receiving(429, 200)

    async def source(request: web.Request) -> web.Response:
        return web.Response(body=_PAYLOAD)

    app.router.add_get("/source", source)
    runner, base = await serve(app)

    try:
        async with ClientSession() as session:
            async with session.get(f"{base}/source") as download:
                response = await asyncio.wait_for(
                    local_requests(retry_max_wait=0, **client_kwargs).post(
                        base, data=download.content
                    ),
                    ENDS_WITHIN_SECONDS,
                )
    finally:
        await runner.cleanup()

    assert response.status == 200
    assert received == [_PAYLOAD, _PAYLOAD]


@pytest.mark.asyncio
@pytest.mark.parametrize("send, part", [("files", "upload"), ("data", "body")])
async def test_a_body_with_no_way_to_be_sent_again_is_refused_when_it_might_be(
    send: str, part: str
):
    """An aiohttp payload over a pipe is used up by its first send. A request
    that might send it again refuses it before sending anything; one that
    sends once streams it."""
    app, received = _receiving(200)
    runner, base = await serve(app)

    try:
        with _pipe(_PAYLOAD) as stream:
            with pytest.raises(
                TypeError, match=f"BufferedReaderPayload {part} can't be sent again"
            ):
                await asyncio.wait_for(
                    local_requests().post(
                        base, **_sent_as(send, BufferedReaderPayload(stream))
                    ),
                    ENDS_WITHIN_SECONDS,
                )
            assert received == []

            response = await asyncio.wait_for(
                local_requests().post(
                    base,
                    allow_redirects=False,
                    **_sent_as(send, BufferedReaderPayload(stream)),
                ),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        await runner.cleanup()

    assert response.status == 200
    assert received == [_expected(send, _PAYLOAD)]


@pytest.mark.asyncio
@pytest.mark.parametrize("send", ["files", "data", "form-dict", "async-iterable"])
async def test_a_body_of_exactly_the_limit_is_held_and_sent(
    send: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(request_body, "MAX_BUFFERED_BODY_BYTES", len(_PAYLOAD))
    app, received = _receiving(200)
    runner, base = await serve(app)

    try:
        with _sent_once(send, _PAYLOAD) as body:
            response = await asyncio.wait_for(
                local_requests().post(base, **body), ENDS_WITHIN_SECONDS
            )
    finally:
        await runner.cleanup()

    assert response.status == 200
    assert received == [
        _PAYLOAD if send == "async-iterable" else _expected(send, _PAYLOAD)
    ]


@pytest.mark.asyncio
async def test_an_async_iterable_body_is_not_run_before_the_url_checks_pass():
    started: list[bool] = []

    async def chunks() -> AsyncIterator[bytes]:
        started.append(True)
        yield _PAYLOAD

    with pytest.raises(ValueError, match="blocked or private IP"):
        await asyncio.wait_for(
            Requests().post("http://127.0.0.1:1/", data=chunks()), ENDS_WITHIN_SECONDS
        )

    assert started == []


class _FailsWithItsOwnTimeout(io.RawIOBase):
    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        raise TimeoutError("the socket under this file timed out")


@pytest.mark.asyncio
async def test_a_timeout_of_the_streams_own_is_not_reported_as_the_requests():
    with pytest.raises(TimeoutError, match="the socket under this file") as raised:
        await asyncio.wait_for(
            local_requests().post(
                "http://127.0.0.1:1/",
                data=_FailsWithItsOwnTimeout(),
                timeout=ClientTimeout(total=30),
            ),
            ENDS_WITHIN_SECONDS,
        )

    assert not isinstance(raised.value, RequestBodyTimeoutError)


@pytest.mark.asyncio
async def test_a_tuple_of_form_fields_goes_as_a_form():
    app, received = _receiving(200)
    runner, base = await serve(app)

    try:
        response = await asyncio.wait_for(
            local_requests().post(base, data=(("a", "1"), ("b", "2"))),
            ENDS_WITHIN_SECONDS,
        )
    finally:
        await runner.cleanup()

    assert response.status == 200
    assert received == [b"a=1&b=2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [[("a", "b", "c")], ["ab"], [("only-a-name",)]],
    ids=["three-items", "a-string", "one-item"],
)
async def test_a_list_item_that_is_not_a_pair_is_refused_before_anything_is_sent(
    fields: list,
):
    app, served = scripted(200)
    runner, base = await serve(app)

    try:
        with pytest.raises(TypeError, match=r"not a \(name, value\) pair"):
            await asyncio.wait_for(
                local_requests().post(base, data=fields), ENDS_WITHIN_SECONDS
            )
    finally:
        await runner.cleanup()

    assert served == []


@contextlib.contextmanager
def _upload_without_a_content_type() -> Iterator[dict]:
    yield {"files": [("file", ("part.bin", BytesIO(_PAYLOAD), ""))]}


@pytest.mark.asyncio
@_HOW
async def test_an_upload_with_no_content_type_goes_as_octet_stream(how: str):
    sends, once = await _carried_twice_and_once(how, _upload_without_a_content_type)

    assert sends == [once, once]
    assert once == [("file", "part.bin", "application/octet-stream", _PAYLOAD)]
