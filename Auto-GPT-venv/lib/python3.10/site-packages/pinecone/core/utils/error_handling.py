#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#
import inspect
from functools import wraps

from urllib3.exceptions import MaxRetryError, ProtocolError

from pinecone import Config, PineconeProtocolError


def validate_and_convert_errors(func):
    @wraps(func)
    def inner_func(*args, **kwargs):
        Config.validate()  # raises exceptions in case of invalid config
        try:
            return func(*args, **kwargs)
        except MaxRetryError as e:
            if isinstance(e.reason, ProtocolError):
                raise PineconeProtocolError(
                    f'Failed to connect to {e.url}; did you specify the correct index name?') from e
            else:
                raise
        except ProtocolError as e:
            raise PineconeProtocolError(f'Failed to connect; did you specify the correct index name?') from e

    # Override signature
    sig = inspect.signature(func)
    inner_func.__signature__ = sig
    return inner_func
