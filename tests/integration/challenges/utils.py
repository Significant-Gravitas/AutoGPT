import contextlib
import functools
import random
from typing import Any, Callable, Dict, Generator, Tuple

import pytest

from autogpt.agent import Agent


def generate_noise(noise_size: int) -> str:
    return "".join(
        random.choices(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            k=noise_size,
        )
    )


def run_multiple_times(times: int) -> Callable:
    """
    Decorator that runs a test function multiple times.

    :param times: The number of times the test function should be executed.
    """

    def decorator(test_func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(test_func)
        def wrapper(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> None:
            for _ in range(times):
                test_func(*args, **kwargs)

        return wrapper

    return decorator


def setup_mock_input(monkeypatch: pytest.MonkeyPatch, cycle_count: int) -> None:
    """
    Sets up the mock input for testing.

    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    :param cycle_count: The number of cycles to mock.
    """
    input_sequence = ["y"] * (cycle_count) + ["EXIT"]

    def input_generator() -> Generator[str, None, None]:
        """
        Creates a generator that yields input strings from the given sequence.
        """
        yield from input_sequence

    gen = input_generator()
    monkeypatch.setattr("builtins.input", lambda _: next(gen))


def run_interaction_loop(
    monkeypatch: pytest.MonkeyPatch, agent: Agent, cycle_count: int
) -> None:
    setup_mock_input(monkeypatch, cycle_count)
    with contextlib.suppress(SystemExit):
        agent.start_interaction_loop()
