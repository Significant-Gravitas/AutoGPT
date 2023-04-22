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
