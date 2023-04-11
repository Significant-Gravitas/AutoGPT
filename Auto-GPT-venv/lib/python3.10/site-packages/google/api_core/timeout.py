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

"""Decorators for applying timeout arguments to functions.

These decorators are used to wrap API methods to apply either a
Deadline-dependent (recommended), constant (DEPRECATED) or exponential
(DEPRECATED) timeout argument.

For example, imagine an API method that can take a while to return results,
such as one that might block until a resource is ready:

.. code-block:: python

    def is_thing_ready(timeout=None):
        response = requests.get('https://example.com/is_thing_ready')
        response.raise_for_status()
        return response.json()

This module allows a function like this to be wrapped so that timeouts are
automatically determined, for example:

.. code-block:: python

    timeout_ = timeout.ExponentialTimeout()
    is_thing_ready_with_timeout = timeout_(is_thing_ready)

    for n in range(10):
        try:
            is_thing_ready_with_timeout({'example': 'data'})
        except:
            pass

In this example the first call to ``is_thing_ready`` will have a relatively
small timeout (like 1 second). If the resource is available and the request
completes quickly, the loop exits. But, if the resource isn't yet available
and the request times out, it'll be retried - this time with a larger timeout.

In the broader context these decorators are typically combined with
:mod:`google.api_core.retry` to implement API methods with a signature that
matches ``api_method(request, timeout=None, retry=None)``.
"""

from __future__ import unicode_literals

import datetime
import functools

from google.api_core import datetime_helpers

_DEFAULT_INITIAL_TIMEOUT = 5.0  # seconds
_DEFAULT_MAXIMUM_TIMEOUT = 30.0  # seconds
_DEFAULT_TIMEOUT_MULTIPLIER = 2.0
# If specified, must be in seconds. If none, deadline is not used in the
# timeout calculation.
_DEFAULT_DEADLINE = None


class TimeToDeadlineTimeout(object):
    """A decorator that decreases timeout set for an RPC based on how much time
    has left till its deadline. The deadline is calculated as
    ``now + initial_timeout`` when this decorator is first called for an rpc.

    In other words this decorator implements deadline semantics in terms of a
    sequence of decreasing timeouts t0 > t1 > t2 ... tn >= 0.

    Args:
        timeout (Optional[float]): the timeout (in seconds) to applied to the
            wrapped function. If `None`, the target function is expected to
            never timeout.
    """

    def __init__(self, timeout=None, clock=datetime_helpers.utcnow):
        self._timeout = timeout
        self._clock = clock

    def __call__(self, func):
        """Apply the timeout decorator.

        Args:
            func (Callable): The function to apply the timeout argument to.
                This function must accept a timeout keyword argument.

        Returns:
            Callable: The wrapped function.
        """

        first_attempt_timestamp = self._clock().timestamp()

        @functools.wraps(func)
        def func_with_timeout(*args, **kwargs):
            """Wrapped function that adds timeout."""

            remaining_timeout = self._timeout
            if remaining_timeout is not None:
                # All calculations are in seconds
                now_timestamp = self._clock().timestamp()

                # To avoid usage of nonlocal but still have round timeout
                # numbers for first attempt (in most cases the only attempt made
                # for an RPC.
                if now_timestamp - first_attempt_timestamp < 0.001:
                    now_timestamp = first_attempt_timestamp

                time_since_first_attempt = now_timestamp - first_attempt_timestamp
                # Avoid setting negative timeout
                kwargs["timeout"] = max(0, self._timeout - time_since_first_attempt)

            return func(*args, **kwargs)

        return func_with_timeout

    def __str__(self):
        return "<TimeToDeadlineTimeout timeout={:.1f}>".format(self._timeout)


class ConstantTimeout(object):
    """A decorator that adds a constant timeout argument.

    DEPRECATED: use ``TimeToDeadlineTimeout`` instead.

    This is effectively equivalent to
    ``functools.partial(func, timeout=timeout)``.

    Args:
        timeout (Optional[float]): the timeout (in seconds) to applied to the
            wrapped function. If `None`, the target function is expected to
            never timeout.
    """

    def __init__(self, timeout=None):
        self._timeout = timeout

    def __call__(self, func):
        """Apply the timeout decorator.

        Args:
            func (Callable): The function to apply the timeout argument to.
                This function must accept a timeout keyword argument.

        Returns:
            Callable: The wrapped function.
        """

        @functools.wraps(func)
        def func_with_timeout(*args, **kwargs):
            """Wrapped function that adds timeout."""
            kwargs["timeout"] = self._timeout
            return func(*args, **kwargs)

        return func_with_timeout

    def __str__(self):
        return "<ConstantTimeout timeout={:.1f}>".format(self._timeout)


def _exponential_timeout_generator(initial, maximum, multiplier, deadline):
    """A generator that yields exponential timeout values.

    Args:
        initial (float): The initial timeout.
        maximum (float): The maximum timeout.
        multiplier (float): The multiplier applied to the timeout.
        deadline (float): The overall deadline across all invocations.

    Yields:
        float: A timeout value.
    """
    if deadline is not None:
        deadline_datetime = datetime_helpers.utcnow() + datetime.timedelta(
            seconds=deadline
        )
    else:
        deadline_datetime = datetime.datetime.max

    timeout = initial
    while True:
        now = datetime_helpers.utcnow()
        yield min(
            # The calculated timeout based on invocations.
            timeout,
            # The set maximum timeout.
            maximum,
            # The remaining time before the deadline is reached.
            float((deadline_datetime - now).seconds),
        )
        timeout = timeout * multiplier


class ExponentialTimeout(object):
    """A decorator that adds an exponentially increasing timeout argument.

    DEPRECATED: the concept of incrementing timeout exponentially has been
    deprecated. Use ``TimeToDeadlineTimeout`` instead.

    This is useful if a function is called multiple times. Each time the
    function is called this decorator will calculate a new timeout parameter
    based on the the number of times the function has been called.

    For example

    .. code-block:: python

    Args:
        initial (float): The initial timeout to pass.
        maximum (float): The maximum timeout for any one call.
        multiplier (float): The multiplier applied to the timeout for each
            invocation.
        deadline (Optional[float]): The overall deadline across all
            invocations. This is used to prevent a very large calculated
            timeout from pushing the overall execution time over the deadline.
            This is especially useful in conjuction with
            :mod:`google.api_core.retry`. If ``None``, the timeouts will not
            be adjusted to accomodate an overall deadline.
    """

    def __init__(
        self,
        initial=_DEFAULT_INITIAL_TIMEOUT,
        maximum=_DEFAULT_MAXIMUM_TIMEOUT,
        multiplier=_DEFAULT_TIMEOUT_MULTIPLIER,
        deadline=_DEFAULT_DEADLINE,
    ):
        self._initial = initial
        self._maximum = maximum
        self._multiplier = multiplier
        self._deadline = deadline

    def with_deadline(self, deadline):
        """Return a copy of this teimout with the given deadline.

        Args:
            deadline (float): The overall deadline across all invocations.

        Returns:
            ExponentialTimeout: A new instance with the given deadline.
        """
        return ExponentialTimeout(
            initial=self._initial,
            maximum=self._maximum,
            multiplier=self._multiplier,
            deadline=deadline,
        )

    def __call__(self, func):
        """Apply the timeout decorator.

        Args:
            func (Callable): The function to apply the timeout argument to.
                This function must accept a timeout keyword argument.

        Returns:
            Callable: The wrapped function.
        """
        timeouts = _exponential_timeout_generator(
            self._initial, self._maximum, self._multiplier, self._deadline
        )

        @functools.wraps(func)
        def func_with_timeout(*args, **kwargs):
            """Wrapped function that adds timeout."""
            kwargs["timeout"] = next(timeouts)
            return func(*args, **kwargs)

        return func_with_timeout

    def __str__(self):
        return (
            "<ExponentialTimeout initial={:.1f}, maximum={:.1f}, "
            "multiplier={:.1f}, deadline={:.1f}>".format(
                self._initial, self._maximum, self._multiplier, self._deadline
            )
        )
