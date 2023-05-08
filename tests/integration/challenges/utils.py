import random
from functools import wraps
from typing import Optional

import pytest


def get_level_to_run(
    user_selected_level: Optional[int],
    level_currently_beaten: Optional[int],
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
        if level_currently_beaten is None:
            pytest.skip(
                "No one has beaten any levels so we cannot run the test in our pipeline"
            )
        # by default we run the level currently beaten.
        return level_currently_beaten
    if user_selected_level > max_level:
        raise ValueError(f"This challenge was not designed to go beyond {max_level}")
    return user_selected_level


def generate_noise(noise_size) -> str:
    return "".join(
        random.choices(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            k=noise_size,
        )
    )


def run_multiple_times(times):
    """
    Decorator that runs a test function multiple times.

    :param times: The number of times the test function should be executed.
    """

    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                test_func(*args, **kwargs)

        return wrapper

    return decorator
