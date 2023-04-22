import os

import pytest


def requires_api_key(env_var):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not os.environ.get(env_var):
                pytest.skip(
                    f"Environment variable '{env_var}' is not set, skipping the test."
                )
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def skip_in_ci():
    def decorator(func):
        def wrapper(*args, **kwargs):
            if os.environ.get("CI"):
                pytest.skip(f"This test doesn't work on GitHub Actions.")
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator
