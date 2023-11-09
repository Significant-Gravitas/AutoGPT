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

"""Transport adapter for urllib3."""

from __future__ import absolute_import

import logging
import os
import warnings

# Certifi is Mozilla's certificate bundle. Urllib3 needs a certificate bundle
# to verify HTTPS requests, and certifi is the recommended and most reliable
# way to get a root certificate bundle. See
# http://urllib3.readthedocs.io/en/latest/user-guide.html\
#   #certificate-verification
# For more details.
try:
    import certifi
except ImportError:  # pragma: NO COVER
    certifi = None  # type: ignore

import six

try:
    import urllib3  # type: ignore
    import urllib3.exceptions  # type: ignore
except ImportError as caught_exc:  # pragma: NO COVER
    six.raise_from(
        ImportError(
            "The urllib3 library is not installed, please install the "
            "urllib3 package to use the urllib3 transport."
        ),
        caught_exc,
    )

from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import service_account

_LOGGER = logging.getLogger(__name__)


class _Response(transport.Response):
    """urllib3 transport response adapter.

    Args:
        response (urllib3.response.HTTPResponse): The raw urllib3 response.
    """

    def __init__(self, response):
        self._response = response

    @property
    def status(self):
        return self._response.status

    @property
    def headers(self):
        return self._response.headers

    @property
    def data(self):
        return self._response.data


class Request(transport.Request):
    """urllib3 request adapter.

    This class is used internally for making requests using various transports
    in a consistent way. If you use :class:`AuthorizedHttp` you do not need
    to construct or use this class directly.

    This class can be useful if you want to manually refresh a
    :class:`~google.auth.credentials.Credentials` instance::

        import google.auth.transport.urllib3
        import urllib3

        http = urllib3.PoolManager()
        request = google.auth.transport.urllib3.Request(http)

        credentials.refresh(request)

    Args:
        http (urllib3.request.RequestMethods): An instance of any urllib3
            class that implements :class:`~urllib3.request.RequestMethods`,
            usually :class:`urllib3.PoolManager`.

    .. automethod:: __call__
    """

    def __init__(self, http):
        self.http = http

    def __call__(
        self, url, method="GET", body=None, headers=None, timeout=None, **kwargs
    ):
        """Make an HTTP request using urllib3.

        Args:
            url (str): The URI to be requested.
            method (str): The HTTP method to use for the request. Defaults
                to 'GET'.
            body (bytes): The payload / body in HTTP request.
            headers (Mapping[str, str]): Request headers.
            timeout (Optional[int]): The number of seconds to wait for a
                response from the server. If not specified or if None, the
                urllib3 default timeout will be used.
            kwargs: Additional arguments passed throught to the underlying
                urllib3 :meth:`urlopen` method.

        Returns:
            google.auth.transport.Response: The HTTP response.

        Raises:
            google.auth.exceptions.TransportError: If any exception occurred.
        """
        # urllib3 uses a sentinel default value for timeout, so only set it if
        # specified.
        if timeout is not None:
            kwargs["timeout"] = timeout

        try:
            _LOGGER.debug("Making request: %s %s", method, url)
            response = self.http.request(
                method, url, body=body, headers=headers, **kwargs
            )
            return _Response(response)
        except urllib3.exceptions.HTTPError as caught_exc:
            new_exc = exceptions.TransportError(caught_exc)
            six.raise_from(new_exc, caught_exc)


def _make_default_http():
    if certifi is not None:
        return urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
    else:
        return urllib3.PoolManager()


def _make_mutual_tls_http(cert, key):
    """Create a mutual TLS HTTP connection with the given client cert and key.
    See https://github.com/urllib3/urllib3/issues/474#issuecomment-253168415

    Args:
        cert (bytes): client certificate in PEM format
        key (bytes): client private key in PEM format

    Returns:
        urllib3.PoolManager: Mutual TLS HTTP connection.

    Raises:
        ImportError: If certifi or pyOpenSSL is not installed.
        OpenSSL.crypto.Error: If the cert or key is invalid.
    """
    import certifi
    from OpenSSL import crypto
    import urllib3.contrib.pyopenssl  # type: ignore

    urllib3.contrib.pyopenssl.inject_into_urllib3()
    ctx = urllib3.util.ssl_.create_urllib3_context()
    ctx.load_verify_locations(cafile=certifi.where())

    pkey = crypto.load_privatekey(crypto.FILETYPE_PEM, key)
    x509 = crypto.load_certificate(crypto.FILETYPE_PEM, cert)

    ctx._ctx.use_certificate(x509)
    ctx._ctx.use_privatekey(pkey)

    http = urllib3.PoolManager(ssl_context=ctx)
    return http


class AuthorizedHttp(urllib3.request.RequestMethods):
    """A urllib3 HTTP class with credentials.

    This class is used to perform requests to API endpoints that require
    authorization::

        from google.auth.transport.urllib3 import AuthorizedHttp

        authed_http = AuthorizedHttp(credentials)

        response = authed_http.request(
            'GET', 'https://www.googleapis.com/storage/v1/b')

    This class implements :class:`urllib3.request.RequestMethods` and can be
    used just like any other :class:`urllib3.PoolManager`.

    The underlying :meth:`urlopen` implementation handles adding the
    credentials' headers to the request and refreshing credentials as needed.

    This class also supports mutual TLS via :meth:`configure_mtls_channel`
    method. In order to use this method, the `GOOGLE_API_USE_CLIENT_CERTIFICATE`
    environment variable must be explicitly set to `true`, otherwise it does
    nothing. Assume the environment is set to `true`, the method behaves in the
    following manner:
    If client_cert_callback is provided, client certificate and private
    key are loaded using the callback; if client_cert_callback is None,
    application default SSL credentials will be used. Exceptions are raised if
    there are problems with the certificate, private key, or the loading process,
    so it should be called within a try/except block.

    First we set the environment variable to `true`, then create an :class:`AuthorizedHttp`
    instance and specify the endpoints::

        regular_endpoint = 'https://pubsub.googleapis.com/v1/projects/{my_project_id}/topics'
        mtls_endpoint = 'https://pubsub.mtls.googleapis.com/v1/projects/{my_project_id}/topics'

        authed_http = AuthorizedHttp(credentials)

    Now we can pass a callback to :meth:`configure_mtls_channel`::

        def my_cert_callback():
            # some code to load client cert bytes and private key bytes, both in
            # PEM format.
            some_code_to_load_client_cert_and_key()
            if loaded:
                return cert, key
            raise MyClientCertFailureException()

        # Always call configure_mtls_channel within a try/except block.
        try:
            is_mtls = authed_http.configure_mtls_channel(my_cert_callback)
        except:
            # handle exceptions.

        if is_mtls:
            response = authed_http.request('GET', mtls_endpoint)
        else:
            response = authed_http.request('GET', regular_endpoint)

    You can alternatively use application default SSL credentials like this::

        try:
            is_mtls = authed_http.configure_mtls_channel()
        except:
            # handle exceptions.

    Args:
        credentials (google.auth.credentials.Credentials): The credentials to
            add to the request.
        http (urllib3.PoolManager): The underlying HTTP object to
            use to make requests. If not specified, a
            :class:`urllib3.PoolManager` instance will be constructed with
            sane defaults.
        refresh_status_codes (Sequence[int]): Which HTTP status codes indicate
            that credentials should be refreshed and the request should be
            retried.
        max_refresh_attempts (int): The maximum number of times to attempt to
            refresh the credentials and retry the request.
        default_host (Optional[str]): A host like "pubsub.googleapis.com".
            This is used when a self-signed JWT is created from service
            account credentials.
    """

    def __init__(
        self,
        credentials,
        http=None,
        refresh_status_codes=transport.DEFAULT_REFRESH_STATUS_CODES,
        max_refresh_attempts=transport.DEFAULT_MAX_REFRESH_ATTEMPTS,
        default_host=None,
    ):
        if http is None:
            self.http = _make_default_http()
            self._has_user_provided_http = False
        else:
            self.http = http
            self._has_user_provided_http = True

        self.credentials = credentials
        self._refresh_status_codes = refresh_status_codes
        self._max_refresh_attempts = max_refresh_attempts
        self._default_host = default_host
        # Request instance used by internal methods (for example,
        # credentials.refresh).
        self._request = Request(self.http)

        # https://google.aip.dev/auth/4111
        # Attempt to use self-signed JWTs when a service account is used.
        if isinstance(self.credentials, service_account.Credentials):
            self.credentials._create_self_signed_jwt(
                "https://{}/".format(self._default_host) if self._default_host else None
            )

        super(AuthorizedHttp, self).__init__()

    def configure_mtls_channel(self, client_cert_callback=None):
        """Configures mutual TLS channel using the given client_cert_callback or
        application default SSL credentials. The behavior is controlled by
        `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable.
        (1) If the environment variable value is `true`, the function returns True
        if the channel is mutual TLS and False otherwise. The `http` provided
        in the constructor will be overwritten.
        (2) If the environment variable is not set or `false`, the function does
        nothing and it always return False.

        Args:
            client_cert_callback (Optional[Callable[[], (bytes, bytes)]]):
                The optional callback returns the client certificate and private
                key bytes both in PEM format.
                If the callback is None, application default SSL credentials
                will be used.

        Returns:
            True if the channel is mutual TLS and False otherwise.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS channel
                creation failed for any reason.
        """
        use_client_cert = os.getenv(
            environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE, "false"
        )
        if use_client_cert != "true":
            return False

        try:
            import OpenSSL
        except ImportError as caught_exc:
            new_exc = exceptions.MutualTLSChannelError(caught_exc)
            six.raise_from(new_exc, caught_exc)

        try:
            found_cert_key, cert, key = transport._mtls_helper.get_client_cert_and_key(
                client_cert_callback
            )

            if found_cert_key:
                self.http = _make_mutual_tls_http(cert, key)
            else:
                self.http = _make_default_http()
        except (
            exceptions.ClientCertError,
            ImportError,
            OpenSSL.crypto.Error,
        ) as caught_exc:
            new_exc = exceptions.MutualTLSChannelError(caught_exc)
            six.raise_from(new_exc, caught_exc)

        if self._has_user_provided_http:
            self._has_user_provided_http = False
            warnings.warn(
                "`http` provided in the constructor is overwritten", UserWarning
            )

        return found_cert_key

    def urlopen(self, method, url, body=None, headers=None, **kwargs):
        """Implementation of urllib3's urlopen."""
        # pylint: disable=arguments-differ
        # We use kwargs to collect additional args that we don't need to
        # introspect here. However, we do explicitly collect the two
        # positional arguments.

        # Use a kwarg for this instead of an attribute to maintain
        # thread-safety.
        _credential_refresh_attempt = kwargs.pop("_credential_refresh_attempt", 0)

        if headers is None:
            headers = self.headers

        # Make a copy of the headers. They will be modified by the credentials
        # and we want to pass the original headers if we recurse.
        request_headers = headers.copy()

        self.credentials.before_request(self._request, method, url, request_headers)

        response = self.http.urlopen(
            method, url, body=body, headers=request_headers, **kwargs
        )

        # If the response indicated that the credentials needed to be
        # refreshed, then refresh the credentials and re-attempt the
        # request.
        # A stored token may expire between the time it is retrieved and
        # the time the request is made, so we may need to try twice.
        # The reason urllib3's retries aren't used is because they
        # don't allow you to modify the request headers. :/
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

            # Recurse. Pass in the original headers, not our modified set.
            return self.urlopen(
                method,
                url,
                body=body,
                headers=headers,
                _credential_refresh_attempt=_credential_refresh_attempt + 1,
                **kwargs
            )

        return response

    # Proxy methods for compliance with the urllib3.PoolManager interface

    def __enter__(self):
        """Proxy to ``self.http``."""
        return self.http.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Proxy to ``self.http``."""
        return self.http.__exit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        if hasattr(self, "http") and self.http is not None:
            self.http.clear()

    @property
    def headers(self):
        """Proxy to ``self.http``."""
        return self.http.headers

    @headers.setter
    def headers(self, value):
        """Proxy to ``self.http``."""
        self.http.headers = value
