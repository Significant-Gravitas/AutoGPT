# Copyright 2020 Google LLC
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

"""Helpers for retrying coroutine functions with exponential back-off.

The :class:`AsyncRetry` decorator shares most functionality and behavior with
:class:`Retry`, but supports coroutine functions. Please refer to description
of :class:`Retry` for more details.

By default, this decorator will retry transient
API errors (see :func:`if_transient_error`). For example:

.. code-block:: python

    @retry_async.AsyncRetry()
    async def call_flaky_rpc():
        return await client.flaky_rpc()

    # Will retry flaky_rpc() if it raises transient API errors.
    result = await call_flaky_rpc()

You can pass a custom predicate to retry on different exceptions, such as
waiting for an eventually consistent item to be available:

.. code-block:: python

    @retry_async.AsyncRetry(predicate=retry_async.if_exception_type(exceptions.NotFound))
    async def check_if_exists():
        return await client.does_thing_exist()

    is_available = await check_if_exists()

Some client library methods apply retry automatically. These methods can accept
a ``retry`` parameter that allows you to configure the behavior:

.. code-block:: python

    my_retry = retry_async.AsyncRetry(deadline=60)
    result = await client.some_method(retry=my_retry)

"""

import asyncio
import datetime
import functools
import logging

from google.api_core import datetime_helpers
from google.api_core import exceptions
from google.api_core.retry import exponential_sleep_generator
from google.api_core.retry import if_exception_type  # noqa: F401
from google.api_core.retry import if_transient_error


_LOGGER = logging.getLogger(__name__)
_DEFAULT_INITIAL_DELAY = 1.0  # seconds
_DEFAULT_MAXIMUM_DELAY = 60.0  # seconds
_DEFAULT_DELAY_MULTIPLIER = 2.0
_DEFAULT_DEADLINE = 60.0 * 2.0  # seconds
_DEFAULT_TIMEOUT = 60.0 * 2.0  # seconds


async def retry_target(
    target, predicate, sleep_generator, timeout=None, on_error=None, **kwargs
):
    """Call a function and retry if it fails.

    This is the lowest-level retry helper. Generally, you'll use the
    higher-level retry helper :class:`Retry`.

    Args:
        target(Callable): The function to call and retry. This must be a
            nullary function - apply arguments with `functools.partial`.
        predicate (Callable[Exception]): A callable used to determine if an
            exception raised by the target should be considered retryable.
            It should return True to retry or False otherwise.
        sleep_generator (Iterable[float]): An infinite iterator that determines
            how long to sleep between retries.
        timeout (float): How long to keep retrying the target, in seconds.
        on_error (Callable[Exception]): A function to call while processing a
            retryable exception.  Any error raised by this function will *not*
            be caught.
        deadline (float): DEPRECATED use ``timeout`` instead. For backward
        compatibility, if set it will override the ``timeout`` parameter.

    Returns:
        Any: the return value of the target function.

    Raises:
        google.api_core.RetryError: If the deadline is exceeded while retrying.
        ValueError: If the sleep generator stops yielding values.
        Exception: If the target raises a method that isn't retryable.
    """

    timeout = kwargs.get("deadline", timeout)

    deadline_dt = (
        (datetime_helpers.utcnow() + datetime.timedelta(seconds=timeout))
        if timeout
        else None
    )

    last_exc = None

    for sleep in sleep_generator:
        try:
            if not deadline_dt:
                return await target()
            else:
                return await asyncio.wait_for(
                    target(),
                    timeout=(deadline_dt - datetime_helpers.utcnow()).total_seconds(),
                )
        # pylint: disable=broad-except
        # This function explicitly must deal with broad exceptions.
        except Exception as exc:
            if not predicate(exc) and not isinstance(exc, asyncio.TimeoutError):
                raise
            last_exc = exc
            if on_error is not None:
                on_error(exc)

        now = datetime_helpers.utcnow()

        if deadline_dt:
            if deadline_dt <= now:
                # Chains the raising RetryError with the root cause error,
                # which helps observability and debugability.
                raise exceptions.RetryError(
                    "Timeout of {:.1f}s exceeded while calling target function".format(
                        timeout
                    ),
                    last_exc,
                ) from last_exc
            else:
                time_to_deadline = (deadline_dt - now).total_seconds()
                sleep = min(time_to_deadline, sleep)

        _LOGGER.debug(
            "Retrying due to {}, sleeping {:.1f}s ...".format(last_exc, sleep)
        )
        await asyncio.sleep(sleep)

    raise ValueError("Sleep generator stopped yielding sleep values.")


class AsyncRetry:
    """Exponential retry decorator for async functions.

    This class is a decorator used to add exponential back-off retry behavior
    to an RPC call.

    Although the default behavior is to retry transient API errors, a
    different predicate can be provided to retry other exceptions.

    Args:
        predicate (Callable[Exception]): A callable that should return ``True``
            if the given exception is retryable.
        initial (float): The minimum a,out of time to delay in seconds. This
            must be greater than 0.
        maximum (float): The maximum amout of time to delay in seconds.
        multiplier (float): The multiplier applied to the delay.
        timeout (float): How long to keep retrying in seconds.
        on_error (Callable[Exception]): A function to call while processing
            a retryable exception. Any error raised by this function will
            *not* be caught.
        deadline (float): DEPRECATED use ``timeout`` instead. If set it will
        override ``timeout`` parameter.
    """

    def __init__(
        self,
        predicate=if_transient_error,
        initial=_DEFAULT_INITIAL_DELAY,
        maximum=_DEFAULT_MAXIMUM_DELAY,
        multiplier=_DEFAULT_DELAY_MULTIPLIER,
        timeout=_DEFAULT_TIMEOUT,
        on_error=None,
        **kwargs
    ):
        self._predicate = predicate
        self._initial = initial
        self._multiplier = multiplier
        self._maximum = maximum
        self._timeout = kwargs.get("deadline", timeout)
        self._deadline = self._timeout
        self._on_error = on_error

    def __call__(self, func, on_error=None):
        """Wrap a callable with retry behavior.

        Args:
            func (Callable): The callable to add retry behavior to.
            on_error (Callable[Exception]): A function to call while processing
                a retryable exception. Any error raised by this function will
                *not* be caught.

        Returns:
            Callable: A callable that will invoke ``func`` with retry
                behavior.
        """
        if self._on_error is not None:
            on_error = self._on_error

        @functools.wraps(func)
        async def retry_wrapped_func(*args, **kwargs):
            """A wrapper that calls target function with retry."""
            target = functools.partial(func, *args, **kwargs)
            sleep_generator = exponential_sleep_generator(
                self._initial, self._maximum, multiplier=self._multiplier
            )
            return await retry_target(
                target,
                self._predicate,
                sleep_generator,
                self._timeout,
                on_error=on_error,
            )

        return retry_wrapped_func

    def _replace(
        self,
        predicate=None,
        initial=None,
        maximum=None,
        multiplier=None,
        timeout=None,
        on_error=None,
    ):
        return AsyncRetry(
            predicate=predicate or self._predicate,
            initial=initial or self._initial,
            maximum=maximum or self._maximum,
            multiplier=multiplier or self._multiplier,
            timeout=timeout or self._timeout,
            on_error=on_error or self._on_error,
        )

    def with_deadline(self, deadline):
        """Return a copy of this retry with the given deadline.
        DEPRECATED: use :meth:`with_timeout` instead.

        Args:
            deadline (float): How long to keep retrying.

        Returns:
            AsyncRetry: A new retry instance with the given deadline.
        """
        return self._replace(timeout=deadline)

    def with_timeout(self, timeout):
        """Return a copy of this retry with the given timeout.

        Args:
            timeout (float): How long to keep retrying, in seconds.

        Returns:
            AsyncRetry: A new retry instance with the given timeout.
        """
        return self._replace(timeout=timeout)

    def with_predicate(self, predicate):
        """Return a copy of this retry with the given predicate.

        Args:
            predicate (Callable[Exception]): A callable that should return
                ``True`` if the given exception is retryable.

        Returns:
            AsyncRetry: A new retry instance with the given predicate.
        """
        return self._replace(predicate=predicate)

    def with_delay(self, initial=None, maximum=None, multiplier=None):
        """Return a copy of this retry with the given delay options.

        Args:
            initial (float): The minimum amout of time to delay. This must
                be greater than 0.
            maximum (float): The maximum amout of time to delay.
            multiplier (float): The multiplier applied to the delay.

        Returns:
            AsyncRetry: A new retry instance with the given predicate.
        """
        return self._replace(initial=initial, maximum=maximum, multiplier=multiplier)

    def __str__(self):
        return (
            "<AsyncRetry predicate={}, initial={:.1f}, maximum={:.1f}, "
            "multiplier={:.1f}, timeout={:.1f}, on_error={}>".format(
                self._predicate,
                self._initial,
                self._maximum,
                self._multiplier,
                self._timeout,
                self._on_error,
            )
        )
