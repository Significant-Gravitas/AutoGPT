# Copyright 2016 Google Inc.
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

"""Transport adapter for httplib2."""

from __future__ import absolute_import

import logging

from google.auth import exceptions
from google.auth import transport
import httplib2
from six.moves import http_client


_LOGGER = logging.getLogger(__name__)
# Properties present in file-like streams / buffers.
_STREAM_PROPERTIES = ("read", "seek", "tell")


class _Response(transport.Response):
    """httplib2 transport response adapter.

    Args:
        response (httplib2.Response): The raw httplib2 response.
        data (bytes): The response body.
    """

    def __init__(self, response, data):
        self._response = response
        self._data = data

    @property
    def status(self):
        """int: The HTTP status code."""
        return self._response.status

    @property
    def headers(self):
        """Mapping[str, str]: The HTTP response headers."""
        return dict(self._response)

    @property
    def data(self):
        """bytes: The response body."""
        return self._data


class Request(transport.Request):
    """httplib2 request adapter.

    This class is used internally for making requests using various transports
    in a consistent way. If you use :class:`AuthorizedHttp` you do not need
    to construct or use this class directly.

    This class can be useful if you want to manually refresh a
    :class:`~google.auth.credentials.Credentials` instance::

        import google_auth_httplib2
        import httplib2

        http = httplib2.Http()
        request = google_auth_httplib2.Request(http)

        credentials.refresh(request)

    Args:
        http (httplib2.Http): The underlying http object to use to make
            requests.

    .. automethod:: __call__
    """

    def __init__(self, http):
        self.http = http

    def __call__(
        self, url, method="GET", body=None, headers=None, timeout=None, **kwargs
    ):
        """Make an HTTP request using httplib2.

        Args:
            url (str): The URI to be requested.
            method (str): The HTTP method to use for the request. Defaults
                to 'GET'.
            body (bytes): The payload / body in HTTP request.
            headers (Mapping[str, str]): Request headers.
            timeout (Optional[int]): The number of seconds to wait for a
                response from the server. This is ignored by httplib2 and will
                issue a warning.
            kwargs: Additional arguments passed throught to the underlying
                :meth:`httplib2.Http.request` method.

        Returns:
            google.auth.transport.Response: The HTTP response.

        Raises:
            google.auth.exceptions.TransportError: If any exception occurred.
        """
        if timeout is not None:
            _LOGGER.warning(
                "httplib2 transport does not support per-request timeout. "
                "Set the timeout when constructing the httplib2.Http instance."
            )

        try:
            _LOGGER.debug("Making request: %s %s", method, url)
            response, data = self.http.request(
                url, method=method, body=body, headers=headers, **kwargs
            )
            return _Response(response, data)
        # httplib2 should catch the lower http error, this is a bug and
        # needs to be fixed there.  Catch the error for the meanwhile.
        except (httplib2.HttpLib2Error, http_client.HTTPException) as exc:
            raise exceptions.TransportError(exc)


def _make_default_http():
    """Returns a default httplib2.Http instance."""
    return httplib2.Http()


class AuthorizedHttp(object):
    """A httplib2 HTTP class with credentials.

    This class is used to perform requests to API endpoints that require
    authorization::

        from google.auth.transport._httplib2 import AuthorizedHttp

        authed_http = AuthorizedHttp(credentials)

        response = authed_http.request(
            'https://www.googleapis.com/storage/v1/b')

    This class implements :meth:`request` in the same way as
    :class:`httplib2.Http` and can usually be used just like any other
    instance of :class:``httplib2.Http`.

    The underlying :meth:`request` implementation handles adding the
    credentials' headers to the request and refreshing credentials as needed.
    """

    def __init__(
        self,
        credentials,
        http=None,
        refresh_status_codes=transport.DEFAULT_REFRESH_STATUS_CODES,
        max_refresh_attempts=transport.DEFAULT_MAX_REFRESH_ATTEMPTS,
    ):
        """
        Args:
            credentials (google.auth.credentials.Credentials): The credentials
                to add to the request.
            http (httplib2.Http): The underlying HTTP object to
                use to make requests. If not specified, a
                :class:`httplib2.Http` instance will be constructed.
            refresh_status_codes (Sequence[int]): Which HTTP status codes
                indicate that credentials should be refreshed and the request
                should be retried.
            max_refresh_attempts (int): The maximum number of times to attempt
                to refresh the credentials and retry the request.
        """

        if http is None:
            http = _make_default_http()

        self.http = http
        self.credentials = credentials
        self._refresh_status_codes = refresh_status_codes
        self._max_refresh_attempts = max_refresh_attempts
        # Request instance used by internal methods (for example,
        # credentials.refresh).
        self._request = Request(self.http)

    def close(self):
        """Calls httplib2's Http.close"""
        self.http.close()

    def request(
        self,
        uri,
        method="GET",
        body=None,
        headers=None,
        redirections=httplib2.DEFAULT_MAX_REDIRECTS,
        connection_type=None,
        **kwargs
    ):
        """Implementation of httplib2's Http.request."""

        _credential_refresh_attempt = kwargs.pop("_credential_refresh_attempt", 0)

        # Make a copy of the headers. They will be modified by the credentials
        # and we want to pass the original headers if we recurse.
        request_headers = headers.copy() if headers is not None else {}

        self.credentials.before_request(self._request, method, uri, request_headers)

        # Check if the body is a file-like stream, and if so, save the body
        # stream position so that it can be restored in case of refresh.
        body_stream_position = None
        if all(getattr(body, stream_prop, None) for stream_prop in _STREAM_PROPERTIES):
            body_stream_position = body.tell()

        # Make the request.
        response, content = self.http.request(
            uri,
            method,
            body=body,
            headers=request_headers,
            redirections=redirections,
            connection_type=connection_type,
            **kwargs
        )

        # If the response indicated that the credentials needed to be
        # refreshed, then refresh the credentials and re-attempt the
        # request.
        # A stored token may expire between the time it is retrieved and
        # the time the request is made, so we may need to try twice.
        if (
            response.status in self._refresh_status_codes
            and _credential_refresh_attempt < self._max_refresh_attempts
        ):

            _LOGGER.info(
                "Refreshing credentials due to a %s response. Attempt %s/%s.",
                response.status,
                _credential_refresh_attempt + 1,
                self._max_refresh_attempts,
            )

            self.credentials.refresh(self._request)

            # Restore the body's stream position if needed.
            if body_stream_position is not None:
                body.seek(body_stream_position)

            # Recurse. Pass in the original headers, not our modified set.
            return self.request(
                uri,
                method,
                body=body,
                headers=headers,
                redirections=redirections,
                connection_type=connection_type,
                _credential_refresh_attempt=_credential_refresh_attempt + 1,
                **kwargs
            )

        return response, content

    def add_certificate(self, key, cert, domain, password=None):
        """Proxy to httplib2.Http.add_certificate."""
        self.http.add_certificate(key, cert, domain, password=password)

    @property
    def connections(self):
        """Proxy to httplib2.Http.connections."""
        return self.http.connections

    @connections.setter
    def connections(self, value):
        """Proxy to httplib2.Http.connections."""
        self.http.connections = value

    @property
    def follow_redirects(self):
        """Proxy to httplib2.Http.follow_redirects."""
        return self.http.follow_redirects

    @follow_redirects.setter
    def follow_redirects(self, value):
        """Proxy to httplib2.Http.follow_redirects."""
        self.http.follow_redirects = value

    @property
    def timeout(self):
        """Proxy to httplib2.Http.timeout."""
        return self.http.timeout

    @timeout.setter
    def timeout(self, value):
        """Proxy to httplib2.Http.timeout."""
        self.http.timeout = value

    @property
    def redirect_codes(self):
        """Proxy to httplib2.Http.redirect_codes."""
        return self.http.redirect_codes

    @redirect_codes.setter
    def redirect_codes(self, value):
        """Proxy to httplib2.Http.redirect_codes."""
        self.http.redirect_codes = value
