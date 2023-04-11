import socket
from time import sleep

from redis.exceptions import ConnectionError, TimeoutError


class Retry:
    """Retry a specific number of times after a failure"""

    def __init__(
        self,
        backoff,
        retries,
        supported_errors=(ConnectionError, TimeoutError, socket.timeout),
    ):
        """
        Initialize a `Retry` object with a `Backoff` object
        that retries a maximum of `retries` times.
        `retries` can be negative to retry forever.
        You can specify the types of supported errors which trigger
        a retry with the `supported_errors` parameter.
        """
        self._backoff = backoff
        self._retries = retries
        self._supported_errors = supported_errors

    def update_supported_errors(self, specified_errors: list):
        """
        Updates the supported errors with the specified error types
        """
        self._supported_errors = tuple(
            set(self._supported_errors + tuple(specified_errors))
        )

    def call_with_retry(self, do, fail):
        """
        Execute an operation that might fail and returns its result, or
        raise the exception that was thrown depending on the `Backoff` object.
        `do`: the operation to call. Expects no argument.
        `fail`: the failure handler, expects the last error that was thrown
        """
        self._backoff.reset()
        failures = 0
        while True:
            try:
                return do()
            except self._supported_errors as error:
                failures += 1
                fail(error)
                if self._retries >= 0 and failures > self._retries:
                    raise error
                backoff = self._backoff.compute(failures)
                if backoff > 0:
                    sleep(backoff)
