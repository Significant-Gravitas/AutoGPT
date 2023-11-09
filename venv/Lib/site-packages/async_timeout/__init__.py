import asyncio
import enum
import sys
import warnings
from types import TracebackType
from typing import Any, Optional, Type


if sys.version_info >= (3, 8):
    from typing import final
else:
    from typing_extensions import final


__version__ = "4.0.2"


__all__ = ("timeout", "timeout_at", "Timeout")


def timeout(delay: Optional[float]) -> "Timeout":
    """timeout context manager.

    Useful in cases when you want to apply timeout logic around block
    of code or in cases when asyncio.wait_for is not suitable. For example:

    >>> async with timeout(0.001):
    ...     async with aiohttp.get('https://github.com') as r:
    ...         await r.text()


    delay - value in seconds or None to disable timeout logic
    """
    loop = _get_running_loop()
    if delay is not None:
        deadline = loop.time() + delay  # type: Optional[float]
    else:
        deadline = None
    return Timeout(deadline, loop)


def timeout_at(deadline: Optional[float]) -> "Timeout":
    """Schedule the timeout at absolute time.

    deadline argument points on the time in the same clock system
    as loop.time().

    Please note: it is not POSIX time but a time with
    undefined starting base, e.g. the time of the system power on.

    >>> async with timeout_at(loop.time() + 10):
    ...     async with aiohttp.get('https://github.com') as r:
    ...         await r.text()


    """
    loop = _get_running_loop()
    return Timeout(deadline, loop)


class _State(enum.Enum):
    INIT = "INIT"
    ENTER = "ENTER"
    TIMEOUT = "TIMEOUT"
    EXIT = "EXIT"


@final
class Timeout:
    # Internal class, please don't instantiate it directly
    # Use timeout() and timeout_at() public factories instead.
    #
    # Implementation note: `async with timeout()` is preferred
    # over `with timeout()`.
    # While technically the Timeout class implementation
    # doesn't need to be async at all,
    # the `async with` statement explicitly points that
    # the context manager should be used from async function context.
    #
    # This design allows to avoid many silly misusages.
    #
    # TimeoutError is raised immadiatelly when scheduled
    # if the deadline is passed.
    # The purpose is to time out as sson as possible
    # without waiting for the next await expression.

    __slots__ = ("_deadline", "_loop", "_state", "_timeout_handler")

    def __init__(
        self, deadline: Optional[float], loop: asyncio.AbstractEventLoop
    ) -> None:
        self._loop = loop
        self._state = _State.INIT

        self._timeout_handler = None  # type: Optional[asyncio.Handle]
        if deadline is None:
            self._deadline = None  # type: Optional[float]
        else:
            self.update(deadline)

    def __enter__(self) -> "Timeout":
        warnings.warn(
            "with timeout() is deprecated, use async with timeout() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._do_enter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        self._do_exit(exc_type)
        return None

    async def __aenter__(self) -> "Timeout":
        self._do_enter()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        self._do_exit(exc_type)
        return None

    @property
    def expired(self) -> bool:
        """Is timeout expired during execution?"""
        return self._state == _State.TIMEOUT

    @property
    def deadline(self) -> Optional[float]:
        return self._deadline

    def reject(self) -> None:
        """Reject scheduled timeout if any."""
        # cancel is maybe better name but
        # task.cancel() raises CancelledError in asyncio world.
        if self._state not in (_State.INIT, _State.ENTER):
            raise RuntimeError(f"invalid state {self._state.value}")
        self._reject()

    def _reject(self) -> None:
        if self._timeout_handler is not None:
            self._timeout_handler.cancel()
            self._timeout_handler = None

    def shift(self, delay: float) -> None:
        """Advance timeout on delay seconds.

        The delay can be negative.

        Raise RuntimeError if shift is called when deadline is not scheduled
        """
        deadline = self._deadline
        if deadline is None:
            raise RuntimeError("cannot shift timeout if deadline is not scheduled")
        self.update(deadline + delay)

    def update(self, deadline: float) -> None:
        """Set deadline to absolute value.

        deadline argument points on the time in the same clock system
        as loop.time().

        If new deadline is in the past the timeout is raised immediatelly.

        Please note: it is not POSIX time but a time with
        undefined starting base, e.g. the time of the system power on.
        """
        if self._state == _State.EXIT:
            raise RuntimeError("cannot reschedule after exit from context manager")
        if self._state == _State.TIMEOUT:
            raise RuntimeError("cannot reschedule expired timeout")
        if self._timeout_handler is not None:
            self._timeout_handler.cancel()
        self._deadline = deadline
        if self._state != _State.INIT:
            self._reschedule()

    def _reschedule(self) -> None:
        assert self._state == _State.ENTER
        deadline = self._deadline
        if deadline is None:
            return

        now = self._loop.time()
        if self._timeout_handler is not None:
            self._timeout_handler.cancel()

        task = _current_task(self._loop)
        if deadline <= now:
            self._timeout_handler = self._loop.call_soon(self._on_timeout, task)
        else:
            self._timeout_handler = self._loop.call_at(deadline, self._on_timeout, task)

    def _do_enter(self) -> None:
        if self._state != _State.INIT:
            raise RuntimeError(f"invalid state {self._state.value}")
        self._state = _State.ENTER
        self._reschedule()

    def _do_exit(self, exc_type: Optional[Type[BaseException]]) -> None:
        if exc_type is asyncio.CancelledError and self._state == _State.TIMEOUT:
            self._timeout_handler = None
            raise asyncio.TimeoutError
        # timeout has not expired
        self._state = _State.EXIT
        self._reject()
        return None

    def _on_timeout(self, task: "asyncio.Task[None]") -> None:
        task.cancel()
        self._state = _State.TIMEOUT
        # drop the reference early
        self._timeout_handler = None


if sys.version_info >= (3, 7):

    def _current_task(loop: asyncio.AbstractEventLoop) -> "Optional[asyncio.Task[Any]]":
        return asyncio.current_task(loop=loop)

else:

    def _current_task(loop: asyncio.AbstractEventLoop) -> "Optional[asyncio.Task[Any]]":
        return asyncio.Task.current_task(loop=loop)


if sys.version_info >= (3, 7):

    def _get_running_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

else:

    def _get_running_loop() -> asyncio.AbstractEventLoop:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            raise RuntimeError("no running event loop")
        return loop
