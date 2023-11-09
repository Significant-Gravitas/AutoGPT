# Copyright 2020, Google LLC
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

"""AsyncIO implementation of the abstract base Future class."""

import asyncio

from google.api_core import exceptions
from google.api_core import retry
from google.api_core import retry_async
from google.api_core.future import base


class _OperationNotComplete(Exception):
    """Private exception used for polling via retry."""

    pass


RETRY_PREDICATE = retry.if_exception_type(
    _OperationNotComplete,
    exceptions.TooManyRequests,
    exceptions.InternalServerError,
    exceptions.BadGateway,
)
DEFAULT_RETRY = retry_async.AsyncRetry(predicate=RETRY_PREDICATE)


class AsyncFuture(base.Future):
    """A Future that polls peer service to self-update.

    The :meth:`done` method should be implemented by subclasses. The polling
    behavior will repeatedly call ``done`` until it returns True.

    .. note::

        Privacy here is intended to prevent the final class from
        overexposing, not to prevent subclasses from accessing methods.

    Args:
        retry (google.api_core.retry.Retry): The retry configuration used
            when polling. This can be used to control how often :meth:`done`
            is polled. Regardless of the retry's ``deadline``, it will be
            overridden by the ``timeout`` argument to :meth:`result`.
    """

    def __init__(self, retry=DEFAULT_RETRY):
        super().__init__()
        self._retry = retry
        self._future = asyncio.get_event_loop().create_future()
        self._background_task = None

    async def done(self, retry=DEFAULT_RETRY):
        """Checks to see if the operation is complete.

        Args:
            retry (google.api_core.retry.Retry): (Optional) How to retry the RPC.

        Returns:
            bool: True if the operation is complete, False otherwise.
        """
        # pylint: disable=redundant-returns-doc, missing-raises-doc
        raise NotImplementedError()

    async def _done_or_raise(self):
        """Check if the future is done and raise if it's not."""
        result = await self.done()
        if not result:
            raise _OperationNotComplete()

    async def running(self):
        """True if the operation is currently running."""
        result = await self.done()
        return not result

    async def _blocking_poll(self, timeout=None):
        """Poll and await for the Future to be resolved.

        Args:
            timeout (int):
                How long (in seconds) to wait for the operation to complete.
                If None, wait indefinitely.
        """
        if self._future.done():
            return

        retry_ = self._retry.with_timeout(timeout)

        try:
            await retry_(self._done_or_raise)()
        except exceptions.RetryError:
            raise asyncio.TimeoutError(
                "Operation did not complete within the designated " "timeout."
            )

    async def result(self, timeout=None):
        """Get the result of the operation.

        Args:
            timeout (int):
                How long (in seconds) to wait for the operation to complete.
                If None, wait indefinitely.

        Returns:
            google.protobuf.Message: The Operation's result.

        Raises:
            google.api_core.GoogleAPICallError: If the operation errors or if
                the timeout is reached before the operation completes.
        """
        await self._blocking_poll(timeout=timeout)
        return self._future.result()

    async def exception(self, timeout=None):
        """Get the exception from the operation.

        Args:
            timeout (int): How long to wait for the operation to complete.
                If None, wait indefinitely.

        Returns:
            Optional[google.api_core.GoogleAPICallError]: The operation's
                error.
        """
        await self._blocking_poll(timeout=timeout)
        return self._future.exception()

    def add_done_callback(self, fn):
        """Add a callback to be executed when the operation is complete.

        If the operation is completed, the callback will be scheduled onto the
        event loop. Otherwise, the callback will be stored and invoked when the
        future is done.

        Args:
            fn (Callable[Future]): The callback to execute when the operation
                is complete.
        """
        if self._background_task is None:
            self._background_task = asyncio.get_event_loop().create_task(
                self._blocking_poll()
            )
        self._future.add_done_callback(fn)

    def set_result(self, result):
        """Set the Future's result."""
        self._future.set_result(result)

    def set_exception(self, exception):
        """Set the Future's exception."""
        self._future.set_exception(exception)
