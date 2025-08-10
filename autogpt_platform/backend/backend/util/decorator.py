import functools
import logging
import os
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Literal,
    ParamSpec,
    Tuple,
    TypeVar,
    overload,
)

from pydantic import BaseModel

from backend.util.logging import TruncatedLogger


class TimingInfo(BaseModel):
    cpu_time: float
    wall_time: float


def _start_measurement() -> Tuple[float, float]:
    return time.time(), os.times()[0] + os.times()[1]


def _end_measurement(
    start_wall_time: float, start_cpu_time: float
) -> Tuple[float, float]:
    end_wall_time = time.time()
    end_cpu_time = os.times()[0] + os.times()[1]
    return end_wall_time - start_wall_time, end_cpu_time - start_cpu_time


P = ParamSpec("P")
T = TypeVar("T")

logger = TruncatedLogger(logging.getLogger(__name__))


def time_measured(
    func: Callable[P, T],
) -> Callable[P, Tuple[TimingInfo, T | BaseException]]:
    """
    Decorator to measure the time taken by a synchronous function to execute.
    """

    @functools.wraps(func)
    def wrapper(
        *args: P.args, **kwargs: P.kwargs
    ) -> Tuple[TimingInfo, T | BaseException]:
        start_wall_time, start_cpu_time = _start_measurement()
        try:
            result = func(*args, **kwargs)
        except BaseException as e:
            result = e
        finally:
            wall_duration, cpu_duration = _end_measurement(
                start_wall_time, start_cpu_time
            )
            timing_info = TimingInfo(cpu_time=cpu_duration, wall_time=wall_duration)
        return timing_info, result

    return wrapper


def async_time_measured(
    func: Callable[P, Awaitable[T]],
) -> Callable[P, Awaitable[Tuple[TimingInfo, T | BaseException]]]:
    """
    Decorator to measure the time taken by an async function to execute.
    """

    @functools.wraps(func)
    async def async_wrapper(
        *args: P.args, **kwargs: P.kwargs
    ) -> Tuple[TimingInfo, T | BaseException]:
        start_wall_time, start_cpu_time = _start_measurement()
        try:
            result = await func(*args, **kwargs)
        except BaseException as e:
            result = e
        finally:
            wall_duration, cpu_duration = _end_measurement(
                start_wall_time, start_cpu_time
            )
            timing_info = TimingInfo(cpu_time=cpu_duration, wall_time=wall_duration)
        return timing_info, result

    return async_wrapper


@overload
def error_logged(
    *, swallow: Literal[True]
) -> Callable[[Callable[P, T]], Callable[P, T | None]]: ...


@overload
def error_logged(
    *, swallow: Literal[False]
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


@overload
def error_logged() -> Callable[[Callable[P, T]], Callable[P, T | None]]: ...


def error_logged(
    *, swallow: bool = True
) -> (
    Callable[[Callable[P, T]], Callable[P, T | None]]
    | Callable[[Callable[P, T]], Callable[P, T]]
):
    """
    Decorator to log any exceptions raised by a function, with optional suppression.

    Args:
        swallow: Whether to suppress the exception (True) or re-raise it (False). Default is True.

    Usage:
        @error_logged()  # Default behavior (swallow errors)
        @error_logged(swallow=False)  # Log and re-raise errors
    """

    def decorator(f: Callable[P, T]) -> Callable[P, T | None]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            try:
                return f(*args, **kwargs)
            except BaseException as e:
                logger.exception(
                    f"Error when calling function {f.__name__} with arguments {args} {kwargs}: {e}"
                )
                if not swallow:
                    raise
                return None

        return wrapper

    return decorator


@overload
def async_error_logged(
    *, swallow: Literal[True]
) -> Callable[
    [Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T | None]]
]: ...


@overload
def async_error_logged(
    *, swallow: Literal[False]
) -> Callable[
    [Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]
]: ...


@overload
def async_error_logged() -> Callable[
    [Callable[P, Coroutine[Any, Any, T]]],
    Callable[P, Coroutine[Any, Any, T | None]],
]: ...


def async_error_logged(*, swallow: bool = True) -> (
    Callable[
        [Callable[P, Coroutine[Any, Any, T]]],
        Callable[P, Coroutine[Any, Any, T | None]],
    ]
    | Callable[
        [Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]
    ]
):
    """
    Decorator to log any exceptions raised by an async function, with optional suppression.

    Args:
        swallow: Whether to suppress the exception (True) or re-raise it (False). Default is True.

    Usage:
        @async_error_logged()  # Default behavior (swallow errors)
        @async_error_logged(swallow=False)  # Log and re-raise errors
    """

    def decorator(
        f: Callable[P, Coroutine[Any, Any, T]],
    ) -> Callable[P, Coroutine[Any, Any, T | None]]:
        @functools.wraps(f)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            try:
                return await f(*args, **kwargs)
            except BaseException as e:
                logger.exception(
                    f"Error when calling async function {f.__name__} with arguments {args} {kwargs}: {e}"
                )
                if not swallow:
                    raise
                return None

        return wrapper

    return decorator
