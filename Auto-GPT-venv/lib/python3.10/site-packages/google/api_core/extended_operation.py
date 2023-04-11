# Copyright 2022 Google LLC
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

"""Futures for extended long-running operations returned from Google Cloud APIs.

These futures can be used to synchronously wait for the result of a
long-running operations using :meth:`ExtendedOperation.result`:

.. code-block:: python

    extended_operation = my_api_client.long_running_method()

    extended_operation.result()

Or asynchronously using callbacks and :meth:`Operation.add_done_callback`:

.. code-block:: python

    extended_operation = my_api_client.long_running_method()

    def my_callback(ex_op):
        print(f"Operation {ex_op.name} completed")

    extended_operation.add_done_callback(my_callback)

"""

import threading

from google.api_core import exceptions
from google.api_core.future import polling


class ExtendedOperation(polling.PollingFuture):
    """An ExtendedOperation future for interacting with a Google API Long-Running Operation.

    Args:
        extended_operation (proto.Message): The initial operation.
        refresh (Callable[[], type(extended_operation)]): A callable that returns
            the latest state of the operation.
        cancel (Callable[[], None]): A callable that tries to cancel the operation.
        polling Optional(google.api_core.retry.Retry): The configuration used
            for polling. This can be used to control how often :meth:`done`
            is polled. If the ``timeout`` argument to :meth:`result` is
            specified it will override the ``polling.timeout`` property.
        retry Optional(google.api_core.retry.Retry): DEPRECATED use ``polling``
            instead. If specified it will override ``polling`` parameter to
            maintain backward compatibility.

    Note: Most long-running API methods use google.api_core.operation.Operation
    This class is a wrapper for a subset of methods that use alternative
    Long-Running Operation (LRO) semantics.

    Note: there is not a concrete type the extended operation must be.
    It MUST have fields that correspond to the following, POSSIBLY WITH DIFFERENT NAMES:
    * name: str
    * status: Union[str, bool, enum.Enum]
    * error_code: int
    * error_message: str
    """

    def __init__(
        self,
        extended_operation,
        refresh,
        cancel,
        polling=polling.DEFAULT_POLLING,
        **kwargs,
    ):
        super().__init__(polling=polling, **kwargs)
        self._extended_operation = extended_operation
        self._refresh = refresh
        self._cancel = cancel
        # Note: the extended operation does not give a good way to indicate cancellation.
        # We make do with manually tracking cancellation and checking for doneness.
        self._cancelled = False
        self._completion_lock = threading.Lock()
        # Invoke in case the operation came back already complete.
        self._handle_refreshed_operation()

    # Note: the following four properties MUST be overridden in a subclass
    # if, and only if, the fields in the corresponding extended operation message
    # have different names.
    #
    # E.g. we have an extended operation class that looks like
    #
    # class MyOperation(proto.Message):
    #     moniker = proto.Field(proto.STRING, number=1)
    #     status_msg = proto.Field(proto.STRING, number=2)
    #     optional http_error_code = proto.Field(proto.INT32, number=3)
    #     optional http_error_msg = proto.Field(proto.STRING, number=4)
    #
    # the ExtendedOperation subclass would provide property overrrides that map
    # to these (poorly named) fields.
    @property
    def name(self):
        return self._extended_operation.name

    @property
    def status(self):
        return self._extended_operation.status

    @property
    def error_code(self):
        return self._extended_operation.error_code

    @property
    def error_message(self):
        return self._extended_operation.error_message

    def __getattr__(self, name):
        return getattr(self._extended_operation, name)

    def done(self, retry=None):
        self._refresh_and_update(retry)
        return self._extended_operation.done

    def cancel(self):
        if self.done():
            return False

        self._cancel()
        self._cancelled = True
        return True

    def cancelled(self):
        # TODO(dovs): there is not currently a good way to determine whether the
        # operation has been cancelled.
        # The best we can do is manually keep track of cancellation
        # and check for doneness.
        if not self._cancelled:
            return False

        self._refresh_and_update()
        return self._extended_operation.done

    def _refresh_and_update(self, retry=None):
        if not self._extended_operation.done:
            self._extended_operation = (
                self._refresh(retry=retry) if retry else self._refresh()
            )
            self._handle_refreshed_operation()

    def _handle_refreshed_operation(self):
        with self._completion_lock:
            if not self._extended_operation.done:
                return

            if self.error_code and self.error_message:
                exception = exceptions.from_http_status(
                    status_code=self.error_code,
                    message=self.error_message,
                    response=self._extended_operation,
                )
                self.set_exception(exception)
            elif self.error_code or self.error_message:
                exception = exceptions.GoogleAPICallError(
                    f"Unexpected error {self.error_code}: {self.error_message}"
                )
                self.set_exception(exception)
            else:
                # Extended operations have no payload.
                self.set_result(None)

    @classmethod
    def make(cls, refresh, cancel, extended_operation, **kwargs):
        """
        Return an instantiated ExtendedOperation (or child) that wraps
        * a refresh callable
        * a cancel callable (can be a no-op)
        * an initial result

        .. note::
            It is the caller's responsibility to set up refresh and cancel
            with their correct request argument.
            The reason for this is that the services that use Extended Operations
            have rpcs that look something like the following:

            // service.proto
            service MyLongService {
                rpc StartLongTask(StartLongTaskRequest) returns (ExtendedOperation) {
                    option (google.cloud.operation_service) = "CustomOperationService";
                }
            }

            service CustomOperationService {
                rpc Get(GetOperationRequest) returns (ExtendedOperation) {
                    option (google.cloud.operation_polling_method) = true;
                }
            }

            Any info needed for the poll, e.g. a name, path params, etc.
            is held in the request, which the initial client method is in a much
            better position to make made because the caller made the initial request.

            TL;DR: the caller sets up closures for refresh and cancel that carry
            the properly configured requests.

        Args:
            refresh (Callable[Optional[Retry]][type(extended_operation)]): A callable that
                returns the latest state of the operation.
            cancel (Callable[][Any]): A callable that tries to cancel the operation
                on a best effort basis.
            extended_operation (Any): The initial response of the long running method.
                See the docstring for ExtendedOperation.__init__ for requirements on
                the type and fields of extended_operation
        """
        return cls(extended_operation, refresh, cancel, **kwargs)
