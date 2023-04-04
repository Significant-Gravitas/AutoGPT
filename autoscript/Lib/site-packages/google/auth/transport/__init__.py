# Copyright 2016 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transport - HTTP client library support.

:mod:`google.auth` is designed to work with various HTTP client libraries such
as urllib3 and requests. In order to work across these libraries with different
interfaces some abstraction is needed.

This module provides two interfaces that are implemented by transport adapters
to support HTTP libraries. :class:`Request` defines the interface expected by
:mod:`google.auth` to make requests. :class:`Response` defines the interface
for the return value of :class:`Request`.
"""

import abc

import six
from six.moves import http_client

TOO_MANY_REQUESTS = 429  # Python 2.7 six is missing this status code.

DEFAULT_RETRYABLE_STATUS_CODES = (
    http_client.INTERNAL_SERVER_ERROR,
    http_client.SERVICE_UNAVAILABLE,
    http_client.REQUEST_TIMEOUT,
    TOO_MANY_REQUESTS,
)
"""Sequence[int]:  HTTP status codes indicating a request can be retried.
"""


DEFAULT_REFRESH_STATUS_CODES = (http_client.UNAUTHORIZED,)
"""Sequence[int]:  Which HTTP status code indicate that credentials should be
refreshed.
"""

DEFAULT_MAX_REFRESH_ATTEMPTS = 2
"""int: How many times to refresh the credentials and retry a request."""


@six.add_metaclass(abc.ABCMeta)
class Response(object):
    """HTTP Response data."""

    @abc.abstractproperty
    def status(self):
        """int: The HTTP status code."""
        raise NotImplementedError("status must be implemented.")

    @abc.abstractproperty
    def headers(self):
        """Mapping[str, str]: The HTTP response headers."""
        raise NotImplementedError("headers must be implemented.")

    @abc.abstractproperty
    def data(self):
        """bytes: The response body."""
        raise NotImplementedError("data must be implemented.")


@six.add_metaclass(abc.ABCMeta)
class Request(object):
    """Interface for a callable that makes HTTP requests.

    Specific transport implementations should provide an implementation of
    this that adapts their specific request / response API.

    .. automethod:: __call__
    """

    @abc.abstractmethod
    def __call__(
        self, url, method="GET", body=None, headers=None, timeout=None, **kwargs
    ):
        """Make an HTTP request.

        Args:
            url (str): The URI to be requested.
            method (str): The HTTP method to use for the request. Defaults
                to 'GET'.
            body (bytes): The payload / body in HTTP request.
            headers (Mapping[str, str]): Request headers.
            timeout (Optional[int]): The number of seconds to wait for a
                response from the server. If not specified or if None, the
                transport-specific default timeout will be used.
            kwargs: Additionally arguments passed on to the transport's
                request method.

        Returns:
            Response: The HTTP response.

        Raises:
            google.auth.exceptions.TransportError: If any exception occurred.
        """
        # pylint: disable=redundant-returns-doc, missing-raises-doc
        # (pylint doesn't play well with abstract docstrings.)
        raise NotImplementedError("__call__ must be implemented.")
