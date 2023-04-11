# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for wrapping low-level gRPC methods with common functionality.

This is used by gapic clients to provide common error mapping, retry, timeout,
pagination, and long-running operations to gRPC methods.
"""

import enum
import functools

from google.api_core import grpc_helpers
from google.api_core.gapic_v1 import client_info
from google.api_core.timeout import TimeToDeadlineTimeout

USE_DEFAULT_METADATA = object()


class _MethodDefault(enum.Enum):
    # Uses enum so that pytype/mypy knows that this is the only possible value.
    # https://stackoverflow.com/a/60605919/101923
    #
    # Literal[_DEFAULT_VALUE] is an alternative, but only added in Python 3.8.
    # https://docs.python.org/3/library/typing.html#typing.Literal
    _DEFAULT_VALUE = object()


DEFAULT = _MethodDefault._DEFAULT_VALUE
"""Sentinel value indicating that a retry or timeout argument was unspecified,
so the default should be used."""


def _is_not_none_or_false(value):
    return value is not None and value is not False


def _apply_decorators(func, decorators):
    """Apply a list of decorators to a given function.

    ``decorators`` may contain items that are ``None`` or ``False`` which will
    be ignored.
    """
    filtered_decorators = filter(_is_not_none_or_false, reversed(decorators))

    for decorator in filtered_decorators:
        func = decorator(func)

    return func


class _GapicCallable(object):
    """Callable that applies retry, timeout, and metadata logic.

    Args:
        target (Callable): The low-level RPC method.
        retry (google.api_core.retry.Retry): The default retry for the
            callable. If ``None``, this callable will not retry by default
        timeout (google.api_core.timeout.Timeout): The default timeout for the
            callable (i.e. duration of time within which an RPC must terminate
            after its start, not to be confused with deadline). If ``None``,
            this callable will not specify a timeout argument to the low-level
            RPC method.
        metadata (Sequence[Tuple[str, str]]): Additional metadata that is
            provided to the RPC method on every invocation. This is merged with
            any metadata specified during invocation. If ``None``, no
            additional metadata will be passed to the RPC method.
    """

    def __init__(self, target, retry, timeout, metadata=None):
        self._target = target
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata

    def __call__(self, *args, timeout=DEFAULT, retry=DEFAULT, **kwargs):
        """Invoke the low-level RPC with retry, timeout, and metadata."""

        if retry is DEFAULT:
            retry = self._retry

        if timeout is DEFAULT:
            timeout = self._timeout

        if isinstance(timeout, (int, float)):
            timeout = TimeToDeadlineTimeout(timeout=timeout)

        # Apply all applicable decorators.
        wrapped_func = _apply_decorators(self._target, [retry, timeout])

        # Add the user agent metadata to the call.
        if self._metadata is not None:
            metadata = kwargs.get("metadata", [])
            # Due to the nature of invocation, None should be treated the same
            # as not specified.
            if metadata is None:
                metadata = []
            metadata = list(metadata)
            metadata.extend(self._metadata)
            kwargs["metadata"] = metadata

        return wrapped_func(*args, **kwargs)


def wrap_method(
    func,
    default_retry=None,
    default_timeout=None,
    client_info=client_info.DEFAULT_CLIENT_INFO,
):
    """Wrap an RPC method with common behavior.

    This applies common error wrapping, retry, and timeout behavior a function.
    The wrapped function will take optional ``retry`` and ``timeout``
    arguments.

    For example::

        import google.api_core.gapic_v1.method
        from google.api_core import retry
        from google.api_core import timeout

        # The original RPC method.
        def get_topic(name, timeout=None):
            request = publisher_v2.GetTopicRequest(name=name)
            return publisher_stub.GetTopic(request, timeout=timeout)

        default_retry = retry.Retry(deadline=60)
        default_timeout = timeout.Timeout(deadline=60)
        wrapped_get_topic = google.api_core.gapic_v1.method.wrap_method(
            get_topic, default_retry)

        # Execute get_topic with default retry and timeout:
        response = wrapped_get_topic()

        # Execute get_topic without doing any retying but with the default
        # timeout:
        response = wrapped_get_topic(retry=None)

        # Execute get_topic but only retry on 5xx errors:
        my_retry = retry.Retry(retry.if_exception_type(
            exceptions.InternalServerError))
        response = wrapped_get_topic(retry=my_retry)

    The way this works is by late-wrapping the given function with the retry
    and timeout decorators. Essentially, when ``wrapped_get_topic()`` is
    called:

    * ``get_topic()`` is first wrapped with the ``timeout`` into
      ``get_topic_with_timeout``.
    * ``get_topic_with_timeout`` is wrapped with the ``retry`` into
      ``get_topic_with_timeout_and_retry()``.
    * The final ``get_topic_with_timeout_and_retry`` is called passing through
      the ``args``  and ``kwargs``.

    The callstack is therefore::

        method.__call__() ->
            Retry.__call__() ->
                Timeout.__call__() ->
                    wrap_errors() ->
                        get_topic()

    Note that if ``timeout`` or ``retry`` is ``None``, then they are not
    applied to the function. For example,
    ``wrapped_get_topic(timeout=None, retry=None)`` is more or less
    equivalent to just calling ``get_topic`` but with error re-mapping.

    Args:
        func (Callable[Any]): The function to wrap. It should accept an
            optional ``timeout`` argument. If ``metadata`` is not ``None``, it
            should accept a ``metadata`` argument.
        default_retry (Optional[google.api_core.Retry]): The default retry
            strategy. If ``None``, the method will not retry by default.
        default_timeout (Optional[google.api_core.Timeout]): The default
            timeout strategy. Can also be specified as an int or float. If
            ``None``, the method will not have timeout specified by default.
        client_info
            (Optional[google.api_core.gapic_v1.client_info.ClientInfo]):
                Client information used to create a user-agent string that's
                passed as gRPC metadata to the method. If unspecified, then
                a sane default will be used. If ``None``, then no user agent
                metadata will be provided to the RPC method.

    Returns:
        Callable: A new callable that takes optional ``retry`` and ``timeout``
            arguments and applies the common error mapping, retry, timeout,
            and metadata behavior to the low-level RPC method.
    """
    func = grpc_helpers.wrap_errors(func)

    if client_info is not None:
        user_agent_metadata = [client_info.to_grpc_metadata()]
    else:
        user_agent_metadata = None

    return functools.wraps(func)(
        _GapicCallable(
            func, default_retry, default_timeout, metadata=user_agent_metadata
        )
    )
