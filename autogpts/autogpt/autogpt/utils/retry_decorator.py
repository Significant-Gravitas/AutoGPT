import inspect
from typing import Optional

import sentry_sdk


def retry(retry_count: int = 3, pass_exception: str = "exception"):
    """Decorator to retry a function multiple times on failure.
    Can pass the exception to the function as a keyword argument."""

    def decorator(func):
        params = inspect.signature(func).parameters

        async def wrapper(*args, **kwargs):
            exception: Optional[Exception] = None
            attempts = 0
            while attempts < retry_count:
                try:
                    if pass_exception in params:
                        kwargs[pass_exception] = exception
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    exception = e
                    sentry_sdk.capture_exception(e)
                    if attempts >= retry_count:
                        raise e

        return wrapper

    return decorator
