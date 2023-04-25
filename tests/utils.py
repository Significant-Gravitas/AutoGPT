import functools
import os
from contextlib import contextmanager

import pytest

from autogpt.config import Config


@contextmanager
def temporary_api_key(config, api_key):
    original_api_key = config.get_openai_api_key()
    config.set_openai_api_key(api_key)
    try:
        yield
    finally:
        config.set_openai_api_key(original_api_key)


def requires_api_key(env_var):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(request, *args, **kwargs):
            use_api_key = request.config.getoption("--use-api-key")
            config = Config()
            if use_api_key:
                return func(request, *args, **kwargs)
            with temporary_api_key(config, "sk-dummy"):
                return func(request, *args, **kwargs)

        return pytest.mark.xfail(strict=False)(wrapper)

    return decorator


def skip_in_ci(test_function):
    return pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="This test doesn't work on GitHub Actions.",
    )(test_function)
