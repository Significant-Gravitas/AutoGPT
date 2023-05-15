import contextlib
import functools
import json
import os
import random
from typing import Any, Callable, Dict, Generator, Optional, Tuple

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


def get_script_directory() -> str:
    return os.path.dirname(os.path.realpath(__file__))


def build_file_path(script_dir: str, filename: str = "current_score.json") -> str:
    return os.path.join(script_dir, filename)


def load_json_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def get_folder_name(func: Callable[..., Any]) -> str:
    return os.path.dirname(os.path.abspath(func.__code__.co_filename)).split(os.sep)[-1]


def get_method_name(func: Callable[..., Any]) -> str:
    return func.__name__.replace("test_", "")


def check_and_update_kwargs(
    test_info: Dict[str, Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    if test_info["current_level_beaten"] == -1:
        pytest.skip("This test has not been unlocked yet.")
    if test_info["max_level"] != 1:
        kwargs["level_to_run"] = test_info["current_level_beaten"]
    return kwargs


def get_folder_key(results: Dict[str, Any], folder_name: str) -> Optional[str]:
    folder_keys = [key for key in results.keys() if folder_name in key]
    return folder_keys[0] if folder_keys else None


def run_test_based_on_current_score(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        if "level_to_run" in kwargs and kwargs["level_to_run"]:
            return func(*args, **kwargs)
        script_dir = get_script_directory()
        file_path = build_file_path(script_dir)
        results = load_json_file(file_path)

        folder_name = get_folder_name(func)
        folder_key = get_folder_key(results, folder_name)

        if folder_key:
            method_name = get_method_name(func)
            if method_name in results[folder_key]:
                test_info = results[folder_key][method_name]
                kwargs = check_and_update_kwargs(test_info, kwargs)

        return func(*args, **kwargs)

    return wrapper
