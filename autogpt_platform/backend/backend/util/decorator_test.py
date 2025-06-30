import time

from backend.util.decorator import error_logged, time_measured


@time_measured
def example_function(a: int, b: int, c: int) -> int:
    time.sleep(0.5)
    return a + b + c


@error_logged
def example_function_with_error(a: int, b: int, c: int) -> int:
    raise ValueError("This is a test error")


def test_timer_decorator():
    info, res = example_function(1, 2, 3)
    assert info.cpu_time >= 0
    assert info.wall_time >= 0.4
    assert res == 6


def test_error_decorator():
    res = example_function_with_error(1, 2, 3)
    assert res is None
