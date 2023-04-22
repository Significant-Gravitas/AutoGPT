import os
import pytest


def requires_openai_api_key(func):
    def wrapper(*args, **kwargs):
        if not os.environ.get('OPENAI_API_KEY'):
            pytest.skip(
                "Environment variable 'OPENAI_API_KEY' is not set, skipping the test."
            )
        else:
            return func(*args, **kwargs)

    return wrapper
