"""Request bodies that go out whole on every send of a request: again after a
throttle status, and on to the next hop of a redirect.

``prepare_body`` decides, before the first send, how each part of a body can
go out again, and ``PreparedBody.for_send`` gives each send its body."""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import threading
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from enum import Enum
from typing import IO, Any

from aiohttp import FormData
from aiohttp.helpers import guess_filename
from aiohttp.payload import Payload
from pydantic import BaseModel

from backend.util.settings import Config

# A file, which aiohttp closes once it has sent it, and an async iterable,
# which runs once, are held in memory so that a retry or a redirect hop can
# send them again: up to the file-size setting, as a download is, for all of
# a body's parts together.
MAX_BUFFERED_BODY_BYTES = Config().max_file_size_mb * 1024 * 1024
_READ_CHUNK_SIZE = 64 * 1024

Upload = IO[bytes] | bytes | bytearray
# A Sequence, not a list: callers build a list of BytesIO uploads, which a
# list of IO[bytes] would not accept, since a list's item type is invariant.
Uploads = Sequence[tuple[str, tuple[str, Upload, str]]]
# The streams aiohttp leaves open once it has sent them, so that the next
# send can rewind them.
_InMemory = io.BytesIO | io.StringIO


class RequestBodyTooLargeError(ValueError):
    """A request body that has to be held in memory is over the limit."""

    def __init__(self, limit: int):
        super().__init__(
            f"Request body is larger than {limit} bytes, the most a request "
            "holds in memory to send again on a retry or a redirect. Pass it as "
            "bytes or a BytesIO, or send it once, with retry_max_attempts=1 and "
            "allow_redirects=False."
        )
        self.limit = limit


class RequestBodyTimeoutError(TimeoutError):
    """A request body took longer to read than the request may take."""

    def __init__(self, timeout: float):
        super().__init__(
            f"The request body took longer to read than the request's {timeout} s "
            "timeout, so nothing was sent."
        )
        self.timeout = timeout


async def prepare_body(
    files: Uploads | None,
    data: Any,
    *,
    sends_again: bool,
    check_url: Callable[[], Awaitable[object]],
    timeout: float | None,
) -> PreparedBody:
    """A request's body, ready for each of its sends.

    A request that sends once gets its body as it is, so a stream goes out
    without being held in memory. One that may send again gets all of its
    body every time:

    - bytes, text, form fields and a ``FormData`` go as they are: aiohttp
      encodes the first three afresh for each send, and sends a ``FormData``'s
      parts again;
    - a ``BytesIO`` or a ``StringIO``, as the body, an upload or a form's
      value, is rewound to where it stood when the request began;
    - any other file is read into memory once, since aiohttp closes a file
      once it has sent it, under the name aiohttp takes a filename and a
      Content-Type from; an async iterable body, which runs once, is read
      into memory too;
    - anything else, such as an aiohttp payload, a ``FormData`` holding a file
      or an async iterable as a form's value, is refused with a ``TypeError``
      before anything is read or sent.

    Reading starts only once ``check_url`` has passed, so a request to a
    blocked URL reads nothing, and a file is read off the event loop. The
    parts held in memory come to at most ``MAX_BUFFERED_BODY_BYTES``, and
    reading them may take at most ``timeout`` seconds, what one send may
    take. aiohttp then gives each send the whole ``timeout`` again, so a
    request may spend up to two of them reading its body and sending it.
    """
    if files is not None and not (data is None or isinstance(data, dict)):
        raise ValueError("When uploading files, 'data' must be a dict of form fields.")
    if not sends_again:
        return PreparedBody.as_it_is(files, data)
    body = PreparedBody.classified(files, data)
    if body.reads_into_memory():
        # A request to a blocked URL must not read a large body first.
        await check_url()
    await body.hold(timeout)
    return body


class PreparedBody:
    """A request's body, part by part: the body itself, or its uploads and
    form fields, or a form's values, each with how it goes out on every send."""

    def __init__(
        self,
        data: _Part,
        fields: list[tuple[Any, _Part]],
        uploads: list[tuple[str, tuple[str, _Part, str]]] | None,
    ):
        self._data = data
        self._fields = fields
        self._uploads = uploads
        self._rewinds: list[tuple[_InMemory, int]] = []

    @classmethod
    def as_it_is(cls, files: Uploads | None, data: Any) -> PreparedBody:
        """A body to send once, as the caller passed it."""
        uploads = [
            (field, (filename, _Part(value=upload, kind=_Kind.AS_IS), mime))
            for field, (filename, upload, mime) in files or []
        ]
        return cls(
            _Part(value=data, kind=_Kind.AS_IS),
            fields=[],
            uploads=None if files is None else uploads,
        )

    @classmethod
    def classified(cls, files: Uploads | None, data: Any) -> PreparedBody:
        """A body that may be sent again, refused before anything is read if
        a part of it can't be."""
        uploads = [
            (field, (filename, _part(upload, in_form=True, what="upload"), mime))
            for field, (filename, upload, mime) in files or []
        ]
        # Alongside uploads, data is the form's text fields, each sent as text.
        body = (
            _Part(value=data, kind=_Kind.AS_IS)
            if files is not None
            else _part(data, in_form=False, what="body")
        )
        fields = [
            (name, _part(value, in_form=True, what="form value"))
            for name, value in (_fields(data) if body.kind is _Kind.FORM else [])
        ]
        return cls(body, fields=fields, uploads=None if files is None else uploads)

    def reads_into_memory(self) -> bool:
        return any(part.kind in (_Kind.READ, _Kind.GATHER) for part in self._parts())

    async def hold(self, timeout: float | None) -> None:
        """Read into memory, in their places, the parts that can't be read
        twice, taking at most *timeout* seconds."""
        self._rewinds = await _hold_within(self._parts(), timeout)

    def for_send(self) -> Any:
        """The ``data`` for one send. The send before it read the body's
        in-memory streams to their end, so they are rewound to where they
        began, and uploads go in a new form, since aiohttp empties a form as
        it encodes it."""
        for stream, start in self._rewinds:
            stream.seek(start)
        if self._uploads is not None:
            form = FormData(quote_fields=False)
            # The form's text fields go first.
            for name, value in (self._data.value or {}).items():
                form.add_field(name, str(value))
            for field, (filename, upload, mime) in self._uploads:
                form.add_field(
                    name=field,
                    value=upload.value,
                    filename=filename,
                    content_type=mime or "application/octet-stream",
                )
            return form
        if any(part.kind is _Kind.READ for _, part in self._fields):
            # A form with files read into memory goes as the list of its
            # fields, which aiohttp encodes as it would the form they came
            # from; any other form goes as the caller passed it.
            return [(name, part.value) for name, part in self._fields]
        return self._data.value

    def _parts(self) -> list[_Part]:
        return [
            *(upload for _, (_, upload, _) in self._uploads or []),
            *(part for _, part in self._fields),
            self._data,
        ]


class _Kind(Enum):
    """How a part of a body can go out on every send."""

    AS_IS = "as is"  # aiohttp encodes it afresh for each send
    REWIND = "rewind"  # a BytesIO or a StringIO, which aiohttp leaves open
    READ = "read"  # any other file, which aiohttp closes once it is sent
    GATHER = "gather"  # an async iterable, which runs once
    FORM = "form"  # a mapping or a list of fields, whose values are parts too
    REFUSE = "refuse"  # no way to send it again


class _Part(BaseModel):
    """A part of a body, the body itself or an upload or a form's value, and
    how it goes out on every send. Reading it into memory replaces its value."""

    value: Any
    kind: _Kind


def _part(value: Any, *, in_form: bool, what: str) -> _Part:
    """*value* as a part of a body, refused if it can't be sent again."""
    kind = _kind(value, in_form=in_form)
    if kind is _Kind.REFUSE:
        raise _cannot_resend(value, what)
    return _Part(value=value, kind=kind)


def _kind(value: Any, *, in_form: bool) -> _Kind:
    """How *value* goes out on every send. With ``_fields``, the one place
    that looks at the type of what a caller passed."""
    if isinstance(value, (io.BytesIO, io.StringIO)):
        return _Kind.REWIND
    if isinstance(value, io.IOBase):
        return _Kind.READ
    if isinstance(value, Payload):
        # aiohttp's own payload. A file's or a stream's is used up by its
        # first send, while bytes, text or JSON would replay; nothing here
        # tells them apart, so all of them are refused.
        return _Kind.REFUSE
    if isinstance(value, AsyncIterable):
        # In a form aiohttp sends one as a part without a filename, where the
        # bytes held in its place would go as a file: refused rather than
        # sent differently.
        return _Kind.REFUSE if in_form else _Kind.GATHER
    if (
        in_form
        or value is None
        or isinstance(value, (bytes, bytearray, memoryview, str))
    ):
        return _Kind.AS_IS
    if isinstance(value, FormData):
        return _form_data_kind(value)
    if isinstance(value, (Mapping, list, tuple)):
        return _Kind.FORM
    return _Kind.REFUSE


def _form_data_kind(form: FormData) -> _Kind:
    """aiohttp encodes a ``FormData``'s parts once and sends those parts again:
    all of a BytesIO or a StringIO each time, but nothing of a file it has
    closed or an async iterable that has run. It offers no way to read a
    form's fields but this private list."""
    kinds = {_kind(value, in_form=True) for _, _, value in form._fields}
    return _Kind.AS_IS if kinds <= {_Kind.AS_IS, _Kind.REWIND} else _Kind.REFUSE


def _fields(form: Any) -> list[tuple[Any, Any]]:
    """A form's fields, as aiohttp reads them from ``data``: a mapping's
    items, or a list's (name, value) pairs, mappings and files, a file named
    as aiohttp names it."""
    if isinstance(form, Mapping):
        return list(form.items())
    fields: list[tuple[Any, Any]] = []
    for item in form:
        if isinstance(item, io.IOBase):
            fields.append((guess_filename(item, "unknown"), item))
        elif isinstance(item, Mapping):
            fields.extend(item.items())
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            fields.append((item[0], item[1]))
        else:
            raise TypeError(
                f"A {type(item).__name__} in a list of form fields is not a "
                "(name, value) pair, a mapping or a file."
            )
    return fields


# What a caller can pass instead of a part that can't be sent again.
_INSTEAD = {
    "body": "bytes, text, a file, a BytesIO, an async iterable, or the form's "
    "fields as a dict or a list of pairs",
    "upload": "bytes, a file or a BytesIO",
    "form value": "text, bytes, a file or a BytesIO",
}


def _cannot_resend(value: Any, what: str) -> TypeError:
    return TypeError(
        f"A {type(value).__name__} {what} can't be sent again by this client, "
        f"on a retry or a redirect. Pass {_INSTEAD[what]} instead, or send it "
        "once, with retry_max_attempts=1 and allow_redirects=False."
    )


async def _hold_within(
    parts: list[_Part], timeout: float | None
) -> list[tuple[_InMemory, int]]:
    """``_hold``, given at most *timeout* seconds."""
    deadline = asyncio.timeout(timeout)
    try:
        async with deadline:
            return await _hold(parts)
    except TimeoutError:
        if timeout is not None and deadline.expired():
            raise RequestBodyTimeoutError(timeout) from None
        raise


async def _hold(parts: list[_Part]) -> list[tuple[_InMemory, int]]:
    """Read each part that can't be read twice into memory, in its place,
    under one limit for them all; and return the in-memory streams each send
    rewinds, with where each begins."""
    rewinds: list[tuple[_InMemory, int]] = []
    room = MAX_BUFFERED_BODY_BYTES
    for part in parts:
        if part.kind is _Kind.REWIND:
            rewinds.append((part.value, part.value.tell()))
        elif part.kind in (_Kind.READ, _Kind.GATHER):
            held, size = await _held(part, room)
            room -= size
            part.value = held
            rewinds.append((held, 0))
    return rewinds


async def _held(part: _Part, room: int) -> tuple[_InMemory, int]:
    """A part's content, written into one stream in memory as it is read, so
    nothing is copied, and its size; refused past *room*."""
    if part.kind is _Kind.GATHER:
        return await _gather_up_to(part.value, room)
    held, size = await _read_in_own_thread(part.value, room)
    # Under the file's own name, from which aiohttp takes a filename and a
    # Content-Type.
    held.name = guess_filename(part.value)
    return held, size


async def _read_in_own_thread(stream: IO[Any], room: int) -> tuple[_InMemory, int]:
    """``_read_up_to`` in a daemon thread of its own, so a slow file holds that
    thread, not the event loop. A stalled read can't be interrupted: in the
    default executor it would keep one of its few workers for good, which
    DNS lookups and file I/O need, and which ``asyncio.run`` and interpreter
    exit wait for. A daemon thread holds none of those, and once this call
    is cancelled it stops at its next chunk instead of reading on."""
    stop = threading.Event()
    outcome: concurrent.futures.Future[tuple[_InMemory, int]] = (
        concurrent.futures.Future()
    )

    def read() -> None:
        if not outcome.set_running_or_notify_cancel():
            return
        try:
            outcome.set_result(_read_up_to(stream, room, stop))
        except BaseException as error:
            outcome.set_exception(error)

    threading.Thread(target=read, daemon=True, name="request-body-read").start()
    try:
        return await asyncio.wrap_future(outcome)
    finally:
        stop.set()


def _read_up_to(
    stream: IO[Any], room: int, stop: threading.Event | None = None
) -> tuple[_InMemory, int]:
    chunk = stream.read(_READ_CHUNK_SIZE)
    held = _in_memory_for(chunk)
    size = 0
    while chunk and (stop is None or not stop.is_set()):
        size += len(chunk)
        if size > room:
            raise RequestBodyTooLargeError(MAX_BUFFERED_BODY_BYTES)
        held.write(chunk)
        chunk = stream.read(_READ_CHUNK_SIZE)
    held.seek(0)
    return held, size


def _in_memory_for(chunk: Any) -> _InMemory:
    """A stream in memory for what a file reads: str for a text file, which
    goes on a ``StringIO``, and bytes for any other, on a ``BytesIO``."""
    return io.StringIO() if isinstance(chunk, str) else io.BytesIO()


async def _gather_up_to(
    chunks: AsyncIterable[bytes], room: int
) -> tuple[io.BytesIO, int]:
    held = io.BytesIO()
    size = 0
    async for chunk in chunks:
        size += len(chunk)
        if size > room:
            raise RequestBodyTooLargeError(MAX_BUFFERED_BODY_BYTES)
        held.write(chunk)
    held.seek(0)
    return held, size
