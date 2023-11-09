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

"""Transport adapter for http.client, for internal use only."""

import logging
import socket

import six
from six.moves import http_client
from six.moves import urllib

from google.auth import exceptions
from google.auth import transport

_LOGGER = logging.getLogger(__name__)


class Response(transport.Response):
    """http.client transport response adapter.

    Args:
        response (http.client.HTTPResponse): The raw http client response.
    """

    def __init__(self, response):
        self._status = response.status
        self._headers = {key.lower(): value for key, value in response.getheaders()}
        self._data = response.read()

    @property
    def status(self):
        return self._status

    @property
    def headers(self):
        return self._headers

    @property
    def data(self):
        return self._data


class Request(transport.Request):
    """http.client transport request adapter."""

    def __call__(
        self, url, method="GET", body=None, headers=None, timeout=None, **kwargs
    ):
        """Make an HTTP request using http.client.

        Args:
            url (str): The URI to be requested.
            method (str): The HTTP method to use for the request. Defaults
                to 'GET'.
            body (bytes): The payload / body in HTTP request.
            headers (Mapping): Request headers.
            timeout (Optional(int)): The number of seconds to wait for a
                response from the server. If not specified or if None, the
                socket global default timeout will be used.
            kwargs: Additional arguments passed throught to the underlying
                :meth:`~http.client.HTTPConnection.request` method.

        Returns:
            Response: The HTTP response.

        Raises:
            google.auth.exceptions.TransportError: If any exception occurred.
        """
        # socket._GLOBAL_DEFAULT_TIMEOUT is the default in http.client.
        if timeout is None:
            timeout = socket._GLOBAL_DEFAULT_TIMEOUT

        # http.client doesn't allow None as the headers argument.
        if headers is None:
            headers = {}

        # http.client needs the host and path parts specified separately.
        parts = urllib.parse.urlsplit(url)
        path = urllib.parse.urlunsplit(
            ("", "", parts.path, parts.query, parts.fragment)
        )

        if parts.scheme != "http":
            raise exceptions.TransportError(
                "http.client transport only supports the http scheme, {}"
                "was specified".format(parts.scheme)
            )

        connection = http_client.HTTPConnection(parts.netloc, timeout=timeout)

        try:
            _LOGGER.debug("Making request: %s %s", method, url)

            connection.request(method, path, body=body, headers=headers, **kwargs)
            response = connection.getresponse()
            return Response(response)

        except (http_client.HTTPException, socket.error) as caught_exc:
            new_exc = exceptions.TransportError(caught_exc)
            six.raise_from(new_exc, caught_exc)

        finally:
            connection.close()
