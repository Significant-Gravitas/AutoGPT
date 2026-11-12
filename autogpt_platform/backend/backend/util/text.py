import asyncio
import atexit
import builtins
import json
import logging
import pickle
import re
import signal
import socket
import subprocess
import sys
import threading
import types
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import update_wrapper
from multiprocessing.connection import Connection
from typing import Any

from jinja2 import BaseLoader, Template
from jinja2.exceptions import TemplateError
from jinja2.sandbox import (
    SandboxedEnvironment,
    SandboxedEscapeFormatter,
    SandboxedFormatter,
)
from markupsafe import Markup

if sys.platform != "win32":
    import resource

logger = logging.getLogger(__name__)

# Resource limits for template rendering
MAX_EXPONENT = 1000  # Max allowed exponent in ** operations
MAX_RANGE = 10_000  # Max items from range()
MAX_SEQUENCE_REPEAT = 10_000  # Max length from sequence * int
TEMPLATE_RENDER_TIMEOUT = 10  # Seconds before render is killed
MAX_FORMAT_FIELD_SIZE = 1_000  # Max width or precision of a formatted field
RENDER_MEMORY_LIMIT = 512 * 2**20  # Bytes a render worker may allocate past its start
MAX_RENDER_OUTPUT = 50_000_000  # Max characters one render may return
MAX_RENDER_WORKERS = 4  # Render worker processes per rendering process
RENDERS_PER_WORKER = 1_000  # Renders before a worker is replaced
WORKER_RETIRE_GROWTH = 64 * 2**20  # Heap growth a render may leave in its worker
WORKER_START_TIMEOUT = 60  # Seconds a new worker may take to answer

REMOVED_FILTERS = frozenset({"pprint", "tojson", "urlize", "xmlattr"})

# The rest of one %-style conversion after its optional "(key)": flags, a
# width and a precision that are digits or "*", a length modifier and the
# conversion type, e.g. "-12.4f" -> ("12", "4", "f"); "*d" -> ("*", "", "d")
_PERCENT_CONVERSION = re.compile(r"[-+ #0]*(\*|\d*)(?:\.(\*|\d*))?[hlL]?([\s\S])")
# The size fields of one str.format() spec, with any fill character, e.g.
# "<12.4f" -> ("12", "4"). A type's own spec language (a datetime's "%Y")
# doesn't match and isn't checked.
_FORMAT_SPEC_SIZES = re.compile(
    r"^(?:[\s\S]?[<>=^])?[-+ ]?z?#?0?(\d*)[,_]?(?:\.(\d+))?[a-zA-Z%]?$"
)


def format_filter_for_jinja2(value, format_string=None):
    if not format_string:
        return value
    _check_percent_format(format_string)
    return format_string % float(value)


def _check_percent_format(format_string: str, args: Any = ()) -> None:
    """Refuse a %-style format string with a field over the size cap.

    A ``*`` width or precision takes its size from ``args``, in the order the
    % operator reads them; a size that isn't there is left for the operator
    to reject.
    """
    values = args if isinstance(args, tuple) else (args,)
    position = 0
    for key, width, precision, conversion in _percent_conversions(format_string):
        sizes: list[str | int | None] = []
        for size in (width, precision):
            if size == "*":
                # Taken even by a literal "%%", so later sizes keep their place
                size = values[position] if position < len(values) else 0
                position += 1
            sizes.append(size if isinstance(size, (str, int)) else 0)
        if conversion != "%":
            position += 1
        _check_field_size(*sizes)


def _percent_conversions(
    format_string: str,
) -> list[tuple[str | None, str, str, str]]:
    """Each %-style conversion as (key, width, precision, type).

    A "(key)" may hold balanced parentheses, as the % operator allows, so
    it's scanned by depth rather than matched up to the first ")".
    """
    conversions = []
    position = format_string.find("%")
    while position != -1:
        position += 1
        key = None
        if format_string.startswith("(", position):
            depth, start = 1, position + 1
            position = start
            while position < len(format_string) and depth:
                depth += {"(": 1, ")": -1}.get(format_string[position], 0)
                position += 1
            key = format_string[start : position - 1]
        rest = _PERCENT_CONVERSION.match(format_string, position)
        if rest is None:
            break
        conversions.append((key, *rest.groups()))
        position = format_string.find("%", rest.end())
    return conversions


def _check_format_spec(format_spec: str) -> None:
    """Refuse a str.format() spec with a field over the size cap."""
    sizes = _FORMAT_SPEC_SIZES.match(format_spec)
    if sizes:
        _check_field_size(sizes.group(1), sizes.group(2))


def _check_field_size(width: str | int | None, precision: str | int | None) -> None:
    # The width and precision decide how much formatting allocates
    if max(abs(int(width or 0)), abs(int(precision or 0))) > MAX_FORMAT_FIELD_SIZE:
        raise OverflowError(
            f"Format field too large (max {MAX_FORMAT_FIELD_SIZE} characters)"
        )


class TextFormatter:
    def __init__(self, autoescape: bool = True):
        self.autoescape = autoescape

    async def format_string(
        self,
        template_str: str,
        values=None,
        *,
        timeout: float | None = TEMPLATE_RENDER_TIMEOUT,
        **kwargs,
    ) -> str:
        """Render a Jinja2 template in a render worker process.

        The worker compiles and renders under ``RENDER_MEMORY_LIMIT`` and is
        killed once ``timeout`` seconds pass, so no template can hold this
        process's memory or time. Waiting for a free worker is not counted
        against ``timeout``.

        Raises ``ValueError`` for anything the caller's template is refused
        for, so the API answers 400 rather than 500.
        """
        context = {**(values or {}), **kwargs}
        return await asyncio.get_running_loop().run_in_executor(
            _render_threads,
            _render_pool.render,
            template_str,
            context,
            self.autoescape,
            timeout,
        )


class _RenderPool:
    """Render workers shared by this process's threads, started on demand."""

    def __init__(self, size: int = MAX_RENDER_WORKERS) -> None:
        self._size = size
        self._idle: list[_RenderWorker] = []
        self._workers = 0
        self._changed = threading.Condition()

    def render(
        self,
        template_str: str,
        context: dict[str, Any],
        autoescape: bool,
        timeout: float | None,
    ) -> str:
        try:
            request = pickle.dumps(
                (template_str, context, autoescape), pickle.HIGHEST_PROTOCOL
            )
        except Exception as e:
            raise ValueError(f"Template values must be plain data: {e}") from e
        worker = self._acquire()
        try:
            answer = worker.render(request, timeout)
        finally:
            self._release(worker)
        return _read_answer(answer)

    def _acquire(self) -> "_RenderWorker":
        with self._changed:
            while True:
                while self._idle:
                    worker = self._idle.pop()
                    if worker.usable:
                        return worker
                    worker.close()
                    self._workers -= 1
                if self._workers < self._size:
                    self._workers += 1
                    break
                self._changed.wait()
        try:
            return _RenderWorker()
        except BaseException:
            with self._changed:
                self._workers -= 1
                self._changed.notify()
            raise

    def close(self) -> None:
        with self._changed:
            for worker in self._idle:
                worker.close()
            self._workers -= len(self._idle)
            self._idle.clear()

    def _release(self, worker: "_RenderWorker") -> None:
        with self._changed:
            if worker.usable:
                self._idle.append(worker)
            else:
                worker.close()
                self._workers -= 1
            self._changed.notify()


class _RenderWorker:
    """One worker process, rendering one template at a time over a socket.

    A fresh interpreter rather than a multiprocessing child, which would
    import the calling program's main module, and with it every module that
    program loaded, before it ran anything of ours.
    """

    def __init__(self) -> None:
        ours, theirs = socket.socketpair()
        # A process-wide socket timeout would leave both ends non-blocking
        ours.setblocking(True)
        theirs.setblocking(True)
        self._process = subprocess.Popen(
            [sys.executable, "-I", "-c", _bootstrap(theirs.fileno())],
            pass_fds=(theirs.fileno(),),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            env={},
        )
        theirs.close()
        self._connection = Connection(ours.detach())
        self.renders = 0
        self.retired = False
        try:
            if not self._connection.poll(WORKER_START_TIMEOUT):
                raise RuntimeError("no answer")
            self._connection.recv_bytes()
        except (RuntimeError, EOFError, OSError) as e:
            self.close()
            raise RuntimeError(f"Template render worker failed to start: {e}") from e

    @property
    def usable(self) -> bool:
        return (
            not self.retired
            and self.renders < RENDERS_PER_WORKER
            and self._process.poll() is None
        )

    def render(self, request: bytes, timeout: float | None) -> bytes:
        self.renders += 1
        try:
            self._connection.send_bytes(request)
            answered = self._connection.poll(timeout)
            answer = self._connection.recv_bytes() if answered else b""
        except (EOFError, OSError) as e:
            self.retire()
            logger.warning(f"Template render worker stopped mid-render: {e!r}")
            raise ValueError("Template rendering failed") from e
        except BaseException:
            # Half a request or answer left on the socket would reach the next caller
            self.retire()
            raise
        if not answered:
            self.retire()
            raise ValueError(
                f"Template rendering timed out after {timeout}s "
                "(expression too complex)"
            )
        if answer[:1] == _RETIRING:
            self.retired = True
        return answer[1:]

    def retire(self) -> None:
        self.retired = True
        self.close()

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        self._connection.close()


def _bootstrap(fd: int) -> str:
    return (
        f"import sys; sys.path[:0] = {sys.path!r}; "
        f"from backend.util.text import _serve_renders; _serve_renders({fd})"
    )


def _read_answer(answer: bytes) -> str:
    """The rendered text, or the worker's error raised here."""
    kind, body = answer[:1], answer[1:].decode("utf-8", "surrogatepass")
    if kind == _TEXT:
        return body
    if kind == _REFUSED:
        raise ValueError(body)
    name, message = json.loads(body)
    # The worker answers in plain text, so only a builtin type is rebuilt
    error_type = getattr(builtins, name, None)
    if isinstance(error_type, type) and issubclass(error_type, Exception):
        try:
            error = error_type(message)
        except TypeError:  # Its constructor wants more than a message
            pass
        else:
            raise error
    raise RuntimeError(f"{name}: {message}")


def _serve_renders(fd: int) -> None:
    """A worker's loop: one request in, one answer out, until the socket closes.

    A render that leaves the heap grown is the worker's last: the allocator
    keeps what it freed, and that would count against the next render's limit.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    connection = Connection(fd)
    started_with = _data_in_use()
    _limit_memory(started_with)
    environments = {False: _build_environment(False), True: _build_environment(True)}
    connection.send_bytes(b"ready")
    while True:
        try:
            answer, out_of_memory = _answer(connection.recv_bytes(), environments)
        except (EOFError, MemoryError):
            # A request too big to receive leaves the rest of it unread
            return
        grown = _data_in_use() > started_with + WORKER_RETIRE_GROWTH
        retiring = out_of_memory or grown
        connection.send_bytes((_RETIRING if retiring else _SERVING) + answer)
        if retiring:
            return


def _answer(
    request: bytes, environments: dict[bool, "_RestrictedEnvironment"]
) -> tuple[bytes, bool]:
    """The answer to one request, and whether the render ran out of memory."""
    try:
        template_str, context, autoescape = pickle.loads(request)
        template = environments[autoescape].from_string(template_str)
        text = _render_capped(template, context)
        return _TEXT + text.encode("utf-8", "surrogatepass"), False
    except Exception as error:
        if _ran_out_of_memory(error):
            limit = RENDER_MEMORY_LIMIT // 2**20
            refusal = f"Template rendering ran out of memory (max {limit} MB)"
            return _REFUSED + refusal.encode(), True
        if isinstance(error, _REFUSALS):
            return _REFUSED + str(error).encode("utf-8", "surrogatepass"), False
        detail = json.dumps([type(error).__name__, str(error)])
        return _ERROR + detail.encode("utf-8", "surrogatepass"), False


def _ran_out_of_memory(error: BaseException) -> bool:
    """Whether a MemoryError is anywhere in the error's chain.

    Jinja rewrites a render's traceback by compiling code, which fails too
    once memory has run out, so what arrives can be a SystemError (3.12) or
    a ValueError (3.11) raised while handling the MemoryError.
    """
    seen: set[int] = set()
    link: BaseException | None = error
    while link is not None and id(link) not in seen:
        if isinstance(link, MemoryError):
            return True
        seen.add(id(link))
        link = link.__cause__ or link.__context__
    return False


def _render_capped(template: Template, context: dict[str, Any]) -> str:
    pieces: list[str] = []
    length = 0
    for piece in template.generate(context):
        length += len(piece)
        if length > MAX_RENDER_OUTPUT:
            raise _OutputTooLarge(
                f"Template output too large (max {MAX_RENDER_OUTPUT} characters)"
            )
        pieces.append(piece)
    return "".join(pieces)


def _limit_memory(in_use: int) -> None:
    # The limit is set only where it is enforced and VmData can be read
    if sys.platform != "linux":
        return
    # On top of what the interpreter and its imports already hold
    _, hard = resource.getrlimit(resource.RLIMIT_DATA)
    limit = in_use + RENDER_MEMORY_LIMIT
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))


def _data_in_use() -> int:
    """Bytes of this process's data segment, which RLIMIT_DATA bounds."""
    if sys.platform != "linux":
        return 0
    with open("/proc/self/status") as status:
        return next(
            int(line.split()[1]) * 1024 for line in status if line.startswith("VmData:")
        )


def _build_environment(autoescape: bool) -> "_RestrictedEnvironment":
    env = _RestrictedEnvironment(loader=BaseLoader(), autoescape=autoescape)
    env.globals.clear()

    # Replace range with a safe capped version
    env.globals["range"] = _safe_range

    # Instead of clearing all filters, just remove potentially unsafe ones
    for name in REMOVED_FILTERS:
        env.filters.pop(name, None)

    env.filters["format"] = format_filter_for_jinja2
    return env


class _OutputTooLarge(Exception):
    """Raised in a worker whose render has emitted more than it may return."""


# A template's own error, answered as ValueError so the API says 400
_REFUSALS = (TemplateError, ArithmeticError, RecursionError, _OutputTooLarge)

# A worker's answer opens with whether it is retiring, then what the rest is
_RETIRING, _SERVING = b"X", b"-"
_TEXT, _REFUSED, _ERROR = b"T", b"R", b"E"

_render_pool = _RenderPool()
# One thread per worker, so renders waiting for one hold no thread of the
# event loop's shared executor
_render_threads = ThreadPoolExecutor(MAX_RENDER_WORKERS, "template-render")
atexit.register(_render_pool.close)


def _safe_range(*args: int) -> range:
    """range() replacement that caps the number of items to prevent DoS."""
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise OverflowError(f"range() too large ({len(r)} items, max {MAX_RANGE})")
    return r


class _RestrictedEnvironment(SandboxedEnvironment):
    """SandboxedEnvironment with computational complexity limits.

    Prevents resource-exhaustion attacks such as ``{{ 999999999**999999999 }}``
    or ``{{ range(999999999) | list }}`` by intercepting dangerous builtins.
    """

    # Tell Jinja2 to route these operators through call_binop()
    intercepted_binops = frozenset(["**", "*", "%"])

    def wrap_str_format(self, value: Any) -> Callable[..., str] | None:
        """``SandboxedEnvironment.wrap_str_format`` with formatters that
        refuse a field over the size cap, so ``str.format`` and
        ``format_map`` get the same limit as the ``format`` filter."""
        if not isinstance(
            value, (types.MethodType, types.BuiltinMethodType)
        ) or value.__name__ not in ("format", "format_map"):
            return None

        f_self: Any = value.__self__
        if not isinstance(f_self, str):
            return None

        str_type: type[str] = type(f_self)
        is_format_map = value.__name__ == "format_map"
        formatter: SandboxedFormatter
        if isinstance(f_self, Markup):
            formatter = _SizeCheckedEscapeFormatter(self, escape=f_self.escape)
        else:
            formatter = _SizeCheckedFormatter(self)
        vformat = formatter.vformat

        def wrapper(*args: Any, **kwargs: Any) -> str:
            if is_format_map:
                if kwargs:
                    raise TypeError("format_map() takes no keyword arguments")
                if len(args) != 1:
                    raise TypeError(
                        f"format_map() takes exactly one argument ({len(args)} given)"
                    )
                kwargs = args[0]
                args = ()
            return str_type(vformat(f_self, args, kwargs))

        return update_wrapper(wrapper, value)

    def call(
        __self,  # noqa: N805 – Jinja2 convention
        __context,
        __obj,
        *args,
        **kwargs,
    ):
        # Intercept pow() to cap the exponent
        if __obj is pow and len(args) >= 2:
            base, exp = args[0], args[1]
            if isinstance(exp, (int, float)) and abs(exp) > MAX_EXPONENT:
                raise OverflowError(f"Exponent too large (max {MAX_EXPONENT})")
            if isinstance(base, (int, float)) and abs(base) > MAX_EXPONENT:
                raise OverflowError(
                    f"Base too large for exponentiation (max {MAX_EXPONENT})"
                )
        return super().call(__context, __obj, *args, **kwargs)

    def call_binop(self, context, operator, left, right):
        # Intercept %-style formatting, of bytes too: a template can make them
        if operator == "%" and isinstance(left, str):
            _check_percent_format(left, right)
        if operator == "%" and isinstance(left, (bytes, bytearray)):
            _check_percent_format(left.decode("latin-1"), right)
        # Intercept the ** (power) operator
        if operator == "**":
            if isinstance(right, (int, float)) and abs(right) > MAX_EXPONENT:
                raise OverflowError(f"Exponent too large (max {MAX_EXPONENT})")
            if isinstance(left, (int, float)) and abs(left) > MAX_EXPONENT:
                raise OverflowError(
                    f"Base too large for exponentiation (max {MAX_EXPONENT})"
                )
        # Intercept sequence repetition via * (strings, lists, tuples)
        if operator == "*":
            if isinstance(left, (str, list, tuple)) and isinstance(right, int):
                if len(left) * right > MAX_SEQUENCE_REPEAT:
                    raise OverflowError(
                        f"Sequence repeat too large (max {MAX_SEQUENCE_REPEAT} items)"
                    )
            if isinstance(right, (str, list, tuple)) and isinstance(left, int):
                if len(right) * left > MAX_SEQUENCE_REPEAT:
                    raise OverflowError(
                        f"Sequence repeat too large (max {MAX_SEQUENCE_REPEAT} items)"
                    )
        return super().call_binop(context, operator, left, right)


class _SizeCheckedFormatter(SandboxedFormatter):
    """A sandboxed str.format() that refuses a field over the size cap.

    Nested sizes such as ``{:{width}}`` are resolved before ``format_field``
    runs, so the check sees the width that would apply.
    """

    def format_field(self, value: Any, format_spec: str) -> Any:
        _check_format_spec(format_spec)
        return super().format_field(value, format_spec)


class _SizeCheckedEscapeFormatter(_SizeCheckedFormatter, SandboxedEscapeFormatter):
    pass


# ---------------------------------------------------------------------------
# CamelCase splitting
# ---------------------------------------------------------------------------

# Map of split forms back to their canonical compound terms.
# Mirrors the frontend exception list in frontend/src/lib/utils.ts.
_CAMELCASE_EXCEPTIONS: dict[str, str] = {
    "Auto GPT": "AutoGPT",
    "Auto Pilot": "AutoPilot",
    "Open AI": "OpenAI",
    "You Tube": "YouTube",
    "Git Hub": "GitHub",
    "Linked In": "LinkedIn",
    "Tik Tok": "TikTok",
    "All Quiet": "AllQuiet",
    "Any Search": "AnySearch",
}

_CAMELCASE_EXCEPTION_RE = re.compile(
    "|".join(re.escape(k) for k in _CAMELCASE_EXCEPTIONS),
)


def split_camelcase(text: str) -> str:
    """Split CamelCase into separate words.

    Uses a single-pass character-by-character algorithm to avoid any
    regex backtracking concerns (guaranteed O(n) time).

    After splitting, known compound terms are restored via an exception
    list (e.g. ``"YouTube"`` stays ``"YouTube"`` instead of becoming
    ``"You Tube"``).  The list mirrors the frontend mapping in
    ``frontend/src/lib/utils.ts``.

    Examples::

        >>> split_camelcase("AITextGeneratorBlock")
        'AI Text Generator Block'
        >>> split_camelcase("OAuth2Block")
        'OAuth2 Block'
        >>> split_camelcase("YouTubeBlock")
        'YouTube Block'
    """
    if len(text) <= 1:
        return text

    parts: list[str] = []
    prev = 0
    for i in range(1, len(text)):
        # Insert split between lowercase/digit and uppercase: "camelCase" -> "camel|Case"
        if (text[i - 1].islower() or text[i - 1].isdigit()) and text[i].isupper():
            parts.append(text[prev:i])
            prev = i
        # Insert split between uppercase run (2+ chars) and uppercase+lowercase:
        # "AIText" -> "AI|Text".  Requires at least 3 consecutive uppercase chars
        # before the lowercase so that the left part keeps 2+ uppercase chars
        # (mirrors the original regex r"([A-Z]{2,})([A-Z][a-z])").
        elif (
            i >= 2
            and text[i - 2].isupper()
            and text[i - 1].isupper()
            and text[i].islower()
            and (i - 1 - prev) >= 2  # left part must retain at least 2 upper chars
        ):
            parts.append(text[prev : i - 1])
            prev = i - 1

    parts.append(text[prev:])
    result = " ".join(parts)

    # Restore known compound terms that should not be split.
    result = _CAMELCASE_EXCEPTION_RE.sub(
        lambda m: _CAMELCASE_EXCEPTIONS[m.group()], result
    )
    return result
