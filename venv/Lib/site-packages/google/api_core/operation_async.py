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

"""AsyncIO futures for long-running operations returned from Google Cloud APIs.

These futures can be used to await for the result of a long-running operation
using :meth:`AsyncOperation.result`:


.. code-block:: python

    operation = my_api_client.long_running_method()
    result = await operation.result()

Or asynchronously using callbacks and :meth:`Operation.add_done_callback`:

.. code-block:: python

    operation = my_api_client.long_running_method()

    def my_callback(future):
        result = await future.result()

    operation.add_done_callback(my_callback)

"""

import functools
import threading

from google.api_core import exceptions
from google.api_core import protobuf_helpers
from google.api_core.future import async_future
from google.longrunning import operations_pb2
from google.rpc import code_pb2


class AsyncOperation(async_future.AsyncFuture):
    """A Future for interacting with a Google API Long-Running Operation.

    Args:
        operation (google.longrunning.operations_pb2.Operation): The
            initial operation.
        refresh (Callable[[], ~.api_core.operation.Operation]): A callable that
            returns the latest state of the operation.
        cancel (Callable[[], None]): A callable that tries to cancel
            the operation.
        result_type (func:`type`): The protobuf type for the operation's
            result.
        metadata_type (func:`type`): The protobuf type for the operation's
            metadata.
        retry (google.api_core.retry.Retry): The retry configuration used
            when polling. This can be used to control how often :meth:`done`
            is polled. Regardless of the retry's ``deadline``, it will be
            overridden by the ``timeout`` argument to :meth:`result`.
    """

    def __init__(
        self,
        operation,
        refresh,
        cancel,
        result_type,
        metadata_type=None,
        retry=async_future.DEFAULT_RETRY,
    ):
        super().__init__(retry=retry)
        self._operation = operation
        self._refresh = refresh
        self._cancel = cancel
        self._result_type = result_type
        self._metadata_type = metadata_type
        self._completion_lock = threading.Lock()
        # Invoke this in case the operation came back already complete.
        self._set_result_from_operation()

    @property
    def operation(self):
        """google.longrunning.Operation: The current long-running operation."""
        return self._operation

    @property
    def metadata(self):
        """google.protobuf.Message: the current operation metadata."""
        if not self._operation.HasField("metadata"):
            return None

        return protobuf_helpers.from_any_pb(
            self._metadata_type, self._operation.metadata
        )

    @classmethod
    def deserialize(cls, payload):
        """Deserialize a ``google.longrunning.Operation`` protocol buffer.

        Args:
            payload (bytes): A serialized operation protocol buffer.

        Returns:
            ~.operations_pb2.Operation: An Operation protobuf object.
        """
        return operations_pb2.Operation.FromString(payload)

    def _set_result_from_operation(self):
        """Set the result or exception from the operation if it is complete."""
        # This must be done in a lock to prevent the async_future thread
        # and main thread from both executing the completion logic
        # at the same time.
        with self._completion_lock:
            # If the operation isn't complete or if the result has already been
            # set, do not call set_result/set_exception again.
            if not self._operation.done or self._future.done():
                return

            if self._operation.HasField("response"):
                response = protobuf_helpers.from_any_pb(
                    self._result_type, self._operation.response
                )
                self.set_result(response)
            elif self._operation.HasField("error"):
                exception = exceptions.GoogleAPICallError(
                    self._operation.error.message,
                    errors=(self._operation.error,),
                    response=self._operation,
                )
                self.set_exception(exception)
            else:
                exception = exceptions.GoogleAPICallError(
                    "Unexpected state: Long-running operation had neither "
                    "response nor error set."
                )
                self.set_exception(exception)

    async def _refresh_and_update(self, retry=async_future.DEFAULT_RETRY):
        """Refresh the operation and update the result if needed.

        Args:
            retry (google.api_core.retry.Retry): (Optional) How to retry the RPC.
        """
        # If the currently cached operation is done, no need to make another
        # RPC as it will not change once done.
        if not self._operation.done:
            self._operation = await self._refresh(retry=retry)
            self._set_result_from_operation()

    async def done(self, retry=async_future.DEFAULT_RETRY):
        """Checks to see if the operation is complete.

        Args:
            retry (google.api_core.retry.Retry): (Optional) How to retry the RPC.

        Returns:
            bool: True if the operation is complete, False otherwise.
        """
        await self._refresh_and_update(retry)
        return self._operation.done

    async def cancel(self):
        """Attempt to cancel the operation.

        Returns:
            bool: True if the cancel RPC was made, False if the operation is
                already complete.
        """
        result = await self.done()
        if result:
            return False
        else:
            await self._cancel()
            return True

    async def cancelled(self):
        """True if the operation was cancelled."""
        await self._refresh_and_update()
        return (
            self._operation.HasField("error")
            and self._operation.error.code == code_pb2.CANCELLED
        )


def from_gapic(operation, operations_client, result_type, grpc_metadata=None, **kwargs):
    """Create an operation future from a gapic client.

    This interacts with the long-running operations `service`_ (specific
    to a given API) via a gapic client.

    .. _service: https://github.com/googleapis/googleapis/blob/\
                 050400df0fdb16f63b63e9dee53819044bffc857/\
                 google/longrunning/operations.proto#L38

    Args:
        operation (google.longrunning.operations_pb2.Operation): The operation.
        operations_client (google.api_core.operations_v1.OperationsClient):
            The operations client.
        result_type (:func:`type`): The protobuf result type.
        grpc_metadata (Optional[List[Tuple[str, str]]]): Additional metadata to pass
            to the rpc.
        kwargs: Keyword args passed into the :class:`Operation` constructor.

    Returns:
        ~.api_core.operation.Operation: The operation future to track the given
            operation.
    """
    refresh = functools.partial(
        operations_client.get_operation, operation.name, metadata=grpc_metadata
    )
    cancel = functools.partial(
        operations_client.cancel_operation, operation.name, metadata=grpc_metadata
    )
    return AsyncOperation(operation, refresh, cancel, result_type, **kwargs)
