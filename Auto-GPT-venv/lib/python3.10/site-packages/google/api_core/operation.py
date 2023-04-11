# Copyright 2016 Google LLC
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

"""Futures for long-running operations returned from Google Cloud APIs.

These futures can be used to synchronously wait for the result of a
long-running operation using :meth:`Operation.result`:


.. code-block:: python

    operation = my_api_client.long_running_method()
    result = operation.result()

Or asynchronously using callbacks and :meth:`Operation.add_done_callback`:

.. code-block:: python

    operation = my_api_client.long_running_method()

    def my_callback(future):
        result = future.result()

    operation.add_done_callback(my_callback)

"""

import functools
import threading

from google.api_core import exceptions
from google.api_core import protobuf_helpers
from google.api_core.future import polling
from google.longrunning import operations_pb2
from google.protobuf import json_format
from google.rpc import code_pb2


class Operation(polling.PollingFuture):
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
        polling (google.api_core.retry.Retry): The configuration used for polling.
            This parameter controls how often :meth:`done` is polled. If the
            ``timeout`` argument is specified in the :meth:`result` method, it will
            override the ``polling.timeout`` property.
        retry (google.api_core.retry.Retry): DEPRECATED: use ``polling`` instead.
            If specified it will override ``polling`` parameter to maintain
            backward compatibility.
    """

    def __init__(
        self,
        operation,
        refresh,
        cancel,
        result_type,
        metadata_type=None,
        polling=polling.DEFAULT_POLLING,
        **kwargs
    ):
        super(Operation, self).__init__(polling=polling, **kwargs)
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
    def deserialize(self, payload):
        """Deserialize a ``google.longrunning.Operation`` protocol buffer.

        Args:
            payload (bytes): A serialized operation protocol buffer.

        Returns:
            ~.operations_pb2.Operation: An Operation protobuf object.
        """
        return operations_pb2.Operation.FromString(payload)

    def _set_result_from_operation(self):
        """Set the result or exception from the operation if it is complete."""
        # This must be done in a lock to prevent the polling thread
        # and main thread from both executing the completion logic
        # at the same time.
        with self._completion_lock:
            # If the operation isn't complete or if the result has already been
            # set, do not call set_result/set_exception again.
            # Note: self._result_set is set to True in set_result and
            # set_exception, in case those methods are invoked directly.
            if not self._operation.done or self._result_set:
                return

            if self._operation.HasField("response"):
                response = protobuf_helpers.from_any_pb(
                    self._result_type, self._operation.response
                )
                self.set_result(response)
            elif self._operation.HasField("error"):
                exception = exceptions.from_grpc_status(
                    status_code=self._operation.error.code,
                    message=self._operation.error.message,
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

    def _refresh_and_update(self, retry=None):
        """Refresh the operation and update the result if needed.

        Args:
            retry (google.api_core.retry.Retry): (Optional) How to retry the RPC.
        """
        # If the currently cached operation is done, no need to make another
        # RPC as it will not change once done.
        if not self._operation.done:
            self._operation = self._refresh(retry=retry) if retry else self._refresh()
            self._set_result_from_operation()

    def done(self, retry=None):
        """Checks to see if the operation is complete.

        Args:
            retry (google.api_core.retry.Retry): (Optional) How to retry the RPC.

        Returns:
            bool: True if the operation is complete, False otherwise.
        """
        self._refresh_and_update(retry)
        return self._operation.done

    def cancel(self):
        """Attempt to cancel the operation.

        Returns:
            bool: True if the cancel RPC was made, False if the operation is
                already complete.
        """
        if self.done():
            return False

        self._cancel()
        return True

    def cancelled(self):
        """True if the operation was cancelled."""
        self._refresh_and_update()
        return (
            self._operation.HasField("error")
            and self._operation.error.code == code_pb2.CANCELLED
        )


def _refresh_http(api_request, operation_name, retry=None):
    """Refresh an operation using a JSON/HTTP client.

    Args:
        api_request (Callable): A callable used to make an API request. This
            should generally be
            :meth:`google.cloud._http.Connection.api_request`.
        operation_name (str): The name of the operation.
        retry (google.api_core.retry.Retry): (Optional) retry policy

    Returns:
        google.longrunning.operations_pb2.Operation: The operation.
    """
    path = "operations/{}".format(operation_name)

    if retry is not None:
        api_request = retry(api_request)

    api_response = api_request(method="GET", path=path)
    return json_format.ParseDict(api_response, operations_pb2.Operation())


def _cancel_http(api_request, operation_name):
    """Cancel an operation using a JSON/HTTP client.

    Args:
        api_request (Callable): A callable used to make an API request. This
            should generally be
            :meth:`google.cloud._http.Connection.api_request`.
        operation_name (str): The name of the operation.
    """
    path = "operations/{}:cancel".format(operation_name)
    api_request(method="POST", path=path)


def from_http_json(operation, api_request, result_type, **kwargs):
    """Create an operation future using a HTTP/JSON client.

    This interacts with the long-running operations `service`_ (specific
    to a given API) via `HTTP/JSON`_.

    .. _HTTP/JSON: https://cloud.google.com/speech/reference/rest/\
            v1beta1/operations#Operation

    Args:
        operation (dict): Operation as a dictionary.
        api_request (Callable): A callable used to make an API request. This
            should generally be
            :meth:`google.cloud._http.Connection.api_request`.
        result_type (:func:`type`): The protobuf result type.
        kwargs: Keyword args passed into the :class:`Operation` constructor.

    Returns:
        ~.api_core.operation.Operation: The operation future to track the given
            operation.
    """
    operation_proto = json_format.ParseDict(operation, operations_pb2.Operation())
    refresh = functools.partial(_refresh_http, api_request, operation_proto.name)
    cancel = functools.partial(_cancel_http, api_request, operation_proto.name)
    return Operation(operation_proto, refresh, cancel, result_type, **kwargs)


def _refresh_grpc(operations_stub, operation_name, retry=None):
    """Refresh an operation using a gRPC client.

    Args:
        operations_stub (google.longrunning.operations_pb2.OperationsStub):
            The gRPC operations stub.
        operation_name (str): The name of the operation.
        retry (google.api_core.retry.Retry): (Optional) retry policy

    Returns:
        google.longrunning.operations_pb2.Operation: The operation.
    """
    request_pb = operations_pb2.GetOperationRequest(name=operation_name)

    rpc = operations_stub.GetOperation
    if retry is not None:
        rpc = retry(rpc)

    return rpc(request_pb)


def _cancel_grpc(operations_stub, operation_name):
    """Cancel an operation using a gRPC client.

    Args:
        operations_stub (google.longrunning.operations_pb2.OperationsStub):
            The gRPC operations stub.
        operation_name (str): The name of the operation.
    """
    request_pb = operations_pb2.CancelOperationRequest(name=operation_name)
    operations_stub.CancelOperation(request_pb)


def from_grpc(operation, operations_stub, result_type, grpc_metadata=None, **kwargs):
    """Create an operation future using a gRPC client.

    This interacts with the long-running operations `service`_ (specific
    to a given API) via gRPC.

    .. _service: https://github.com/googleapis/googleapis/blob/\
                 050400df0fdb16f63b63e9dee53819044bffc857/\
                 google/longrunning/operations.proto#L38

    Args:
        operation (google.longrunning.operations_pb2.Operation): The operation.
        operations_stub (google.longrunning.operations_pb2.OperationsStub):
            The operations stub.
        result_type (:func:`type`): The protobuf result type.
        grpc_metadata (Optional[List[Tuple[str, str]]]): Additional metadata to pass
            to the rpc.
        kwargs: Keyword args passed into the :class:`Operation` constructor.

    Returns:
        ~.api_core.operation.Operation: The operation future to track the given
            operation.
    """
    refresh = functools.partial(
        _refresh_grpc, operations_stub, operation.name, metadata=grpc_metadata
    )
    cancel = functools.partial(
        _cancel_grpc, operations_stub, operation.name, metadata=grpc_metadata
    )
    return Operation(operation, refresh, cancel, result_type, **kwargs)


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
    return Operation(operation, refresh, cancel, result_type, **kwargs)
