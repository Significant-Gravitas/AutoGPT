import functools
import logging
import os
import time
from typing import Any, Awaitable, Callable, Coroutine, ParamSpec, Tuple, TypeVar

from pydantic import BaseModel


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

logger = logging.getLogger(__name__)


def time_measured(func: Callable[P, T]) -> Callable[P, Tuple[TimingInfo, T]]:
    """
    Decorator to measure the time taken by a synchronous function to execute.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Tuple[TimingInfo, T]:
        start_wall_time, start_cpu_time = _start_measurement()
        try:
            result = func(*args, **kwargs)
        finally:
            wall_duration, cpu_duration = _end_measurement(
                start_wall_time, start_cpu_time
            )
            timing_info = TimingInfo(cpu_time=cpu_duration, wall_time=wall_duration)
        return timing_info, result

    return wrapper


def async_time_measured(
    func: Callable[P, Awaitable[T]],
) -> Callable[P, Awaitable[Tuple[TimingInfo, T]]]:
    """
    Decorator to measure the time taken by an async function to execute.
    """

    @functools.wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Tuple[TimingInfo, T]:
        start_wall_time, start_cpu_time = _start_measurement()
        try:
            result = await func(*args, **kwargs)
        finally:
            wall_duration, cpu_duration = _end_measurement(
                start_wall_time, start_cpu_time
            )
            timing_info = TimingInfo(cpu_time=cpu_duration, wall_time=wall_duration)
        return timing_info, result

    return async_wrapper


def error_logged(func: Callable[P, T]) -> Callable[P, T | None]:
    """
    Decorator to suppress and log any exceptions raised by a function.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(
                f"Error when calling function {func.__name__} with arguments {args} {kwargs}: {e}"
            )

    return wrapper


def async_error_logged(
    func: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T | None]]:
    """
    Decorator to suppress and log any exceptions raised by an async function.
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.exception(
                f"Error when calling async function {func.__name__} with arguments {args} {kwargs}: {e}"
            )

    return wrapper
