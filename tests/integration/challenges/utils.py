import contextlib
import random
from functools import wraps
from typing import Any, Callable, Dict, Generator, Optional, Tuple

import pytest

from autogpt.agent import Agent


def get_level_to_run(
    user_selected_level: int,
    level_currently_beaten: int,
    max_level: int,
) -> int:
    """
    Determines the appropriate level to run for a challenge, based on user-selected level, level currently beaten, and maximum level.

    Args:
        user_selected_level (int | None): The level selected by the user. If not provided, the level currently beaten is used.
        level_currently_beaten (int | None): The highest level beaten so far. If not provided, the test will be skipped.
        max_level (int): The maximum level allowed for the challenge.

    Returns:
        int: The level to run for the challenge.

    Raises:
        ValueError: If the user-selected level is greater than the maximum level allowed.
    """
    if user_selected_level is None:
        if level_currently_beaten == -1:
            pytest.skip(
                "No one has beaten any levels so we cannot run the test in our pipeline"
            )
        # by default we run the level currently beaten.
        return level_currently_beaten
    if user_selected_level > max_level:
        raise ValueError(f"This challenge was not designed to go beyond {max_level}")
    return user_selected_level


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
        @wraps(test_func)
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
