# Copyright 2017, Google LLC
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

"""Abstract and helper bases for Future implementations."""

import abc
import concurrent.futures

from google.api_core import exceptions
from google.api_core import retry as retries
from google.api_core.future import _helpers
from google.api_core.future import base


class _OperationNotComplete(Exception):
    """Private exception used for polling via retry."""

    pass


# DEPRECATED as it conflates RPC retry and polling concepts into one.
# Use POLLING_PREDICATE instead to configure polling.
RETRY_PREDICATE = retries.if_exception_type(
    _OperationNotComplete,
    exceptions.TooManyRequests,
    exceptions.InternalServerError,
    exceptions.BadGateway,
    exceptions.ServiceUnavailable,
)

# DEPRECATED: use DEFAULT_POLLING to configure LRO polling logic. Construct
# Retry object using its default values as a baseline for any custom retry logic
# (not to be confused with polling logic).
DEFAULT_RETRY = retries.Retry(predicate=RETRY_PREDICATE)

# POLLING_PREDICATE is supposed to poll only on _OperationNotComplete.
# Any RPC-specific errors (like ServiceUnavailable) will be handled
# by retry logic (not to be confused with polling logic) which is triggered for
# every polling RPC independently of polling logic but within its context.
POLLING_PREDICATE = retries.if_exception_type(
    _OperationNotComplete,
)

# Default polling configuration
DEFAULT_POLLING = retries.Retry(
    predicate=POLLING_PREDICATE,
    initial=1.0,  # seconds
    maximum=20.0,  # seconds
    multiplier=1.5,
    timeout=900,  # seconds
)


class PollingFuture(base.Future):
    """A Future that needs to poll some service to check its status.

    The :meth:`done` method should be implemented by subclasses. The polling
    behavior will repeatedly call ``done`` until it returns True.

    The actuall polling logic is encapsulated in :meth:`result` method. See
    documentation for that method for details on how polling works.

    .. note::

        Privacy here is intended to prevent the final class from
        overexposing, not to prevent subclasses from accessing methods.

    Args:
        polling (google.api_core.retry.Retry): The configuration used for polling.
            This parameter controls how often :meth:`done` is polled. If the
            ``timeout`` argument is specified in :meth:`result` method it will
            override the ``polling.timeout`` property.
        retry (google.api_core.retry.Retry): DEPRECATED use ``polling`` instead.
            If set, it will override ``polling`` paremeter for backward
            compatibility.
    """

    _DEFAULT_VALUE = object()

    def __init__(self, polling=DEFAULT_POLLING, **kwargs):
        super(PollingFuture, self).__init__()
        self._polling = kwargs.get("retry", polling)
        self._result = None
        self._exception = None
        self._result_set = False
        """bool: Set to True when the result has been set via set_result or
        set_exception."""
        self._polling_thread = None
        self._done_callbacks = []

    @abc.abstractmethod
    def done(self, retry=None):
        """Checks to see if the operation is complete.

        Args:
            retry (google.api_core.retry.Retry): (Optional) How to retry the
                polling RPC (to not be confused with polling configuration. See
                the documentation for :meth:`result` for details).

        Returns:
            bool: True if the operation is complete, False otherwise.
        """
        # pylint: disable=redundant-returns-doc, missing-raises-doc
        raise NotImplementedError()

    def _done_or_raise(self, retry=None):
        """Check if the future is done and raise if it's not."""
        if not self.done(retry=retry):
            raise _OperationNotComplete()

    def running(self):
        """True if the operation is currently running."""
        return not self.done()

    def _blocking_poll(self, timeout=_DEFAULT_VALUE, retry=None, polling=None):
        """Poll and wait for the Future to be resolved."""

        if self._result_set:
            return

        polling = polling or self._polling
        if timeout is not PollingFuture._DEFAULT_VALUE:
            polling = polling.with_timeout(timeout)

        try:
            polling(self._done_or_raise)(retry=retry)
        except exceptions.RetryError:
            raise concurrent.futures.TimeoutError(
                f"Operation did not complete within the designated timeout of "
                f"{polling.timeout} seconds."
            )

    def result(self, timeout=_DEFAULT_VALUE, retry=None, polling=None):
        """Get the result of the operation.

        This method will poll for operation status periodically, blocking if
        necessary. If you just want to make sure that this method does not block
        for more than X seconds and you do not care about the nitty-gritty of
        how this method operates, just call it with ``result(timeout=X)``. The
        other parameters are for advanced use only.

        Every call to this method is controlled by the following three
        parameters, each of which has a specific, distinct role, even though all three
        may look very similar: ``timeout``, ``retry`` and ``polling``. In most
        cases users do not need to specify any custom values for any of these
        parameters and may simply rely on default ones instead.

        If you choose to specify custom parameters, please make sure you've
        read the documentation below carefully.

        First, please check :class:`google.api_core.retry.Retry`
        class documentation for the proper definition of timeout and deadline
        terms and for the definition the three different types of timeouts.
        This class operates in terms of Retry Timeout and Polling Timeout. It
        does not let customizing RPC timeout and the user is expected to rely on
        default behavior for it.

        The roles of each argument of this method are as follows:

        ``timeout`` (int): (Optional) The Polling Timeout as defined in
        :class:`google.api_core.retry.Retry`. If the operation does not complete
        within this timeout an exception will be thrown. This parameter affects
        neither Retry Timeout nor RPC Timeout.

        ``retry`` (google.api_core.retry.Retry): (Optional) How to retry the
        polling RPC. The ``retry.timeout`` property of this parameter is the
        Retry Timeout as defined in :class:`google.api_core.retry.Retry`.
        This parameter defines ONLY how the polling RPC call is retried
        (i.e. what to do if the RPC we used for polling returned an error). It
        does NOT define how the polling is done (i.e. how frequently and for
        how long to call the polling RPC); use the ``polling`` parameter for that.
        If a polling RPC throws and error and retrying it fails, the whole
        future fails with the corresponding exception. If you want to tune which
        server response error codes are not fatal for operation polling, use this
        parameter to control that (``retry.predicate`` in particular).

        ``polling`` (google.api_core.retry.Retry): (Optional) How often and
        for how long to call the polling RPC periodically (i.e. what to do if
        a polling rpc returned successfully but its returned result indicates
        that the long running operation is not completed yet, so we need to
        check it again at some point in future). This parameter does NOT define
        how to retry each individual polling RPC in case of an error; use the
        ``retry`` parameter for that. The ``polling.timeout`` of this parameter
        is Polling Timeout as defined in as defined in
        :class:`google.api_core.retry.Retry`.

        For each of the arguments, there are also default values in place, which
        will be used if a user does not specify their own. The default values
        for the three parameters are not to be confused with the default values
        for the corresponding arguments in this method (those serve as "not set"
        markers for the resolution logic).

        If ``timeout`` is provided (i.e.``timeout is not _DEFAULT VALUE``; note
        the ``None`` value means "infinite timeout"), it will be used to control
        the actual Polling Timeout. Otherwise, the ``polling.timeout`` value
        will be used instead (see below for how the ``polling`` config itself
        gets resolved). In other words, this parameter  effectively overrides
        the ``polling.timeout`` value if specified. This is so to preserve
        backward compatibility.

        If ``retry`` is provided (i.e. ``retry is not None``) it will be used to
        control retry behavior for the polling RPC and the ``retry.timeout``
        will determine the Retry Timeout. If not provided, the
        polling RPC will be called with whichever default retry config was
        specified for the polling RPC at the moment of the construction of the
        polling RPC's client. For example, if the polling RPC is
        ``operations_client.get_operation()``, the ``retry`` parameter will be
        controlling its retry behavior (not polling  behavior) and, if not
        specified, that specific method (``operations_client.get_operation()``)
        will be retried according to the default retry config provided during
        creation of ``operations_client`` client instead. This argument exists
        mainly for backward compatibility; users are very unlikely to ever need
        to set this parameter explicitly.

        If ``polling`` is provided (i.e. ``polling is not None``), it will be used
        to controll the overall polling behavior and ``polling.timeout`` will
        controll Polling Timeout unless it is overridden by ``timeout`` parameter
        as described above. If not provided, the``polling`` parameter specified
        during construction of this future (the ``polling`` argument in the
        constructor) will be used instead. Note: since the ``timeout`` argument may
        override ``polling.timeout`` value, this parameter should be viewed as
        coupled with the ``timeout`` parameter as described above.

        Args:
            timeout (int): (Optional) How long (in seconds) to wait for the
                operation to complete. If None, wait indefinitely.
            retry (google.api_core.retry.Retry): (Optional) How to retry the
                polling RPC. This defines ONLY how the polling RPC call is
                retried (i.e. what to do if the RPC we used for polling returned
                an error). It does  NOT define how the polling is done (i.e. how
                frequently and for how long to call the polling RPC).
            polling (google.api_core.retry.Retry): (Optional) How often and
                for how long to call polling RPC periodically. This parameter
                does NOT define how to retry each individual polling RPC call
                (use the ``retry`` parameter for that).

        Returns:
            google.protobuf.Message: The Operation's result.

        Raises:
            google.api_core.GoogleAPICallError: If the operation errors or if
                the timeout is reached before the operation completes.
        """

        self._blocking_poll(timeout=timeout, retry=retry, polling=polling)

        if self._exception is not None:
            # pylint: disable=raising-bad-type
            # Pylint doesn't recognize that this is valid in this case.
            raise self._exception

        return self._result

    def exception(self, timeout=_DEFAULT_VALUE):
        """Get the exception from the operation, blocking if necessary.

        See the documentation for the :meth:`result` method for details on how
        this method operates, as both ``result`` and this method rely on the
        exact same polling logic. The only difference is that this method does
        not accept ``retry`` and ``polling`` arguments but relies on the default ones
        instead.

        Args:
            timeout (int): How long to wait for the operation to complete.
            If None, wait indefinitely.

        Returns:
            Optional[google.api_core.GoogleAPICallError]: The operation's
                error.
        """
        self._blocking_poll(timeout=timeout)
        return self._exception

    def add_done_callback(self, fn):
        """Add a callback to be executed when the operation is complete.

        If the operation is not already complete, this will start a helper
        thread to poll for the status of the operation in the background.

        Args:
            fn (Callable[Future]): The callback to execute when the operation
                is complete.
        """
        if self._result_set:
            _helpers.safe_invoke_callback(fn, self)
            return

        self._done_callbacks.append(fn)

        if self._polling_thread is None:
            # The polling thread will exit on its own as soon as the operation
            # is done.
            self._polling_thread = _helpers.start_daemon_thread(
                target=self._blocking_poll
            )

    def _invoke_callbacks(self, *args, **kwargs):
        """Invoke all done callbacks."""
        for callback in self._done_callbacks:
            _helpers.safe_invoke_callback(callback, *args, **kwargs)

    def set_result(self, result):
        """Set the Future's result."""
        self._result = result
        self._result_set = True
        self._invoke_callbacks(self)

    def set_exception(self, exception):
        """Set the Future's exception."""
        self._exception = exception
        self._result_set = True
        self._invoke_callbacks(self)
