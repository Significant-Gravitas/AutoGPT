import functools
import os

import pytest


def requires_api_key(env_var):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.environ.get(env_var):
                pytest.skip(
                    f"Environment variable '{env_var}' is not set, skipping the test."
                )
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def skip_in_ci(test_function):
    return pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="This test doesn't work on GitHub Actions.",
    )(test_function)


def skip_if_dumb_llm(test_function):
    """Skip a test if a very smart LLM is not available"""
    return pytest.mark.skipif(
        os.environ.get("SMART_LLM_MODEL") != "gpt-4",
        reason="This test only works with GPT-4",
    )(test_function)
