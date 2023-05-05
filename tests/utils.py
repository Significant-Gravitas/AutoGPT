import functools
import os
from contextlib import contextmanager

import pytest

from autogpt.config import Config


@contextmanager
def dummy_openai_api_key():
    # even when we record the VCR cassettes, openAI wants an API key
    config = Config()
    original_api_key = config.openai_api_key
    config.set_openai_api_key("sk-dummy")

    try:
        yield
    finally:
        config.set_openai_api_key(original_api_key)


def requires_api_key(env_var):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.environ.get(env_var) and env_var == "OPENAI_API_KEY":
                with dummy_openai_api_key():
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def skip_in_ci(test_function):
    return pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="This test doesn't work on GitHub Actions.",
    )(test_function)


def get_workspace_file_path(workspace, file_name):
    return str(workspace.get_path(file_name))
