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

"""Transport adapter for Requests."""

from __future__ import absolute_import

import functools
import logging
import numbers
import os
import time

try:
    import requests
except ImportError as caught_exc:  # pragma: NO COVER
    import six

    six.raise_from(
        ImportError(
            "The requests library is not installed, please install the "
            "requests package to use the requests transport."
        ),
        caught_exc,
    )
import requests.adapters  # pylint: disable=ungrouped-imports
import requests.exceptions  # pylint: disable=ungrouped-imports
from requests.packages.urllib3.util.ssl_ import (  # type: ignore
    create_urllib3_context,
)  # pylint: disable=ungrouped-imports
import six  # pylint: disable=ungrouped-imports

from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
import google.auth.transport._mtls_helper
from google.oauth2 import service_account

_LOGGER = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120  # in seconds


class _Response(transport.Response):
    """Requests transport response adapter.

    Args:
        response (requests.Response): The raw Requests response.
    """

    def __init__(self, response):
        self._response = response

    @property
    def status(self):
        return self._response.status_code

    @property
    def headers(self):
        return self._response.headers

    @property
    def data(self):
        return self._response.content


class TimeoutGuard(object):
    """A context manager raising an error if the suite execution took too long.

    Args:
        timeout (Union[None, Union[float, Tuple[float, float]]]):
            The maximum number of seconds a suite can run without the context
            manager raising a timeout exception on exit. If passed as a tuple,
            the smaller of the values is taken as a timeout. If ``None``, a
            timeout error is never raised.
        timeout_error_type (Optional[Exception]):
            The type of the error to raise on timeout. Defaults to
            :class:`requests.exceptions.Timeout`.
    """

    def __init__(self, timeout, timeout_error_type=requests.exceptions.Timeout):
        self._timeout = timeout
        self.remaining_timeout = timeout
        self._timeout_error_type = timeout_error_type

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value:
            return  # let the error bubble up automatically

        if self._timeout is None:
            return  # nothing to do, the timeout was not specified

        elapsed = time.time() - self._start
        deadline_hit = False

        if isinstance(self._timeout, numbers.Number):
            self.remaining_timeout = self._timeout - elapsed
            deadline_hit = self.remaining_timeout <= 0
        else:
            self.remaining_timeout = tuple(x - elapsed for x in self._timeout)
            deadline_hit = min(self.remaining_timeout) <= 0

        if deadline_hit:
            raise self._timeout_error_type()


class Request(transport.Request):
    """Requests request adapter.

    This class is used internally for making requests using various transports
    in a consistent way. If you use :class:`AuthorizedSession` you do not need
    to construct or use this class directly.

    This class can be useful if you want to manually refresh a
    :class:`~google.auth.credentials.Credentials` instance::

        import google.auth.transport.requests
        import requests

        request = google.auth.transport.requests.Request()

        credentials.refresh(request)

    Args:
        session (requests.Session): An instance :class:`requests.Session` used
            to make HTTP requests. If not specified, a session will be created.

    .. automethod:: __call__
    """

    def __init__(self, session=None):
        if not session:
            session = requests.Session()

        self.session = session

    def __del__(self):
        try:
            if hasattr(self, "session") and self.session is not None:
                self.session.close()
        except TypeError:
            # NOTE: For certain Python binary built, the queue.Empty exception
            # might not be considered a normal Python exception causing
            # TypeError.
            pass

    def __call__(
        self,
        url,
        method="GET",
        body=None,
        headers=None,
        timeout=_DEFAULT_TIMEOUT,
        **kwargs
    ):
        """Make an HTTP request using requests.

        Args:
            url (str): The URI to be requested.
            method (str): The HTTP method to use for the request. Defaults
                to 'GET'.
            body (bytes): The payload or body in HTTP request.
            headers (Mapping[str, str]): Request headers.
            timeout (Optional[int]): The number of seconds to wait for a
                response from the server. If not specified or if None, the
                requests default timeout will be used.
            kwargs: Additional arguments passed through to the underlying
                requests :meth:`~requests.Session.request` method.

        Returns:
            google.auth.transport.Response: The HTTP response.

        Raises:
            google.auth.exceptions.TransportError: If any exception occurred.
        """
        try:
            _LOGGER.debug("Making request: %s %s", method, url)
            response = self.session.request(
                method, url, data=body, headers=headers, timeout=timeout, **kwargs
            )
            return _Response(response)
        except requests.exceptions.RequestException as caught_exc:
            new_exc = exceptions.TransportError(caught_exc)
            six.raise_from(new_exc, caught_exc)


class _MutualTlsAdapter(requests.adapters.HTTPAdapter):
    """
    A TransportAdapter that enables mutual TLS.

    Args:
        cert (bytes): client certificate in PEM format
        key (bytes): client private key in PEM format

    Raises:
        ImportError: if certifi or pyOpenSSL is not installed
        OpenSSL.crypto.Error: if client cert or key is invalid
    """

    def __init__(self, cert, key):
        import certifi
        from OpenSSL import crypto
        import urllib3.contrib.pyopenssl  # type: ignore

        urllib3.contrib.pyopenssl.inject_into_urllib3()

        pkey = crypto.load_privatekey(crypto.FILETYPE_PEM, key)
        x509 = crypto.load_certificate(crypto.FILETYPE_PEM, cert)

        ctx_poolmanager = create_urllib3_context()
        ctx_poolmanager.load_verify_locations(cafile=certifi.where())
        ctx_poolmanager._ctx.use_certificate(x509)
        ctx_poolmanager._ctx.use_privatekey(pkey)
        self._ctx_poolmanager = ctx_poolmanager

        ctx_proxymanager = create_urllib3_context()
        ctx_proxymanager.load_verify_locations(cafile=certifi.where())
        ctx_proxymanager._ctx.use_certificate(x509)
        ctx_proxymanager._ctx.use_privatekey(pkey)
        self._ctx_proxymanager = ctx_proxymanager

        super(_MutualTlsAdapter, self).__init__()

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ctx_poolmanager
        super(_MutualTlsAdapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ctx_proxymanager
        return super(_MutualTlsAdapter, self).proxy_manager_for(*args, **kwargs)


class _MutualTlsOffloadAdapter(requests.adapters.HTTPAdapter):
    """
    A TransportAdapter that enables mutual TLS and offloads the client side
    signing operation to the signing library.

    Args:
        enterprise_cert_file_path (str): the path to a enterprise cert JSON
            file. The file should contain the following field:

                {
                    "libs": {
                        "signer_library": "...",
                        "offload_library": "..."
                    }
                }

    Raises:
        ImportError: if certifi or pyOpenSSL is not installed
        google.auth.exceptions.MutualTLSChannelError: If mutual TLS channel
            creation failed for any reason.
    """

    def __init__(self, enterprise_cert_file_path):
        import certifi
        import urllib3.contrib.pyopenssl

        from google.auth.transport import _custom_tls_signer

        # Call inject_into_urllib3 to activate certificate checking. See the
        # following links for more info:
        # (1) doc: https://github.com/urllib3/urllib3/blob/cb9ebf8aac5d75f64c8551820d760b72b619beff/src/urllib3/contrib/pyopenssl.py#L31-L32
        # (2) mTLS example: https://github.com/urllib3/urllib3/issues/474#issuecomment-253168415
        urllib3.contrib.pyopenssl.inject_into_urllib3()

        self.signer = _custom_tls_signer.CustomTlsSigner(enterprise_cert_file_path)
        self.signer.load_libraries()
        self.signer.set_up_custom_key()

        poolmanager = create_urllib3_context()
        poolmanager.load_verify_locations(cafile=certifi.where())
        self.signer.attach_to_ssl_context(poolmanager)
        self._ctx_poolmanager = poolmanager

        proxymanager = create_urllib3_context()
        proxymanager.load_verify_locations(cafile=certifi.where())
        self.signer.attach_to_ssl_context(proxymanager)
        self._ctx_proxymanager = proxymanager

        super(_MutualTlsOffloadAdapter, self).__init__()

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ctx_poolmanager
        super(_MutualTlsOffloadAdapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ctx_proxymanager
        return super(_MutualTlsOffloadAdapter, self).proxy_manager_for(*args, **kwargs)


class AuthorizedSession(requests.Session):
    """A Requests Session class with credentials.

    This class is used to perform requests to API endpoints that require
    authorization::

        from google.auth.transport.requests import AuthorizedSession

        authed_session = AuthorizedSession(credentials)

        response = authed_session.request(
            'GET', 'https://www.googleapis.com/storage/v1/b')


    The underlying :meth:`request` implementation handles adding the
    credentials' headers to the request and refreshing credentials as needed.

    This class also supports mutual TLS via :meth:`configure_mtls_channel`
    method. In order to use this method, the `GOOGLE_API_USE_CLIENT_CERTIFICATE`
    environment variable must be explicitly set to ``true``, otherwise it does
    nothing. Assume the environment is set to ``true``, the method behaves in the
    following manner:

    If client_cert_callback is provided, client certificate and private
    key are loaded using the callback; if client_cert_callback is None,
    application default SSL credentials will be used. Exceptions are raised if
    there are problems with the certificate, private key, or the loading process,
    so it should be called within a try/except block.

    First we set the environment variable to ``true``, then create an :class:`AuthorizedSession`
    instance and specify the endpoints::

        regular_endpoint = 'https://pubsub.googleapis.com/v1/projects/{my_project_id}/topics'
        mtls_endpoint = 'https://pubsub.mtls.googleapis.com/v1/projects/{my_project_id}/topics'

        authed_session = AuthorizedSession(credentials)

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
            authed_session.configure_mtls_channel(my_cert_callback)
        except:
            # handle exceptions.

        if authed_session.is_mtls:
            response = authed_session.request('GET', mtls_endpoint)
        else:
            response = authed_session.request('GET', regular_endpoint)


    You can alternatively use application default SSL credentials like this::

        try:
            authed_session.configure_mtls_channel()
        except:
            # handle exceptions.

    Args:
        credentials (google.auth.credentials.Credentials): The credentials to
            add to the request.
        refresh_status_codes (Sequence[int]): Which HTTP status codes indicate
            that credentials should be refreshed and the request should be
            retried.
        max_refresh_attempts (int): The maximum number of times to attempt to
            refresh the credentials and retry the request.
        refresh_timeout (Optional[int]): The timeout value in seconds for
            credential refresh HTTP requests.
        auth_request (google.auth.transport.requests.Request):
            (Optional) An instance of
            :class:`~google.auth.transport.requests.Request` used when
            refreshing credentials. If not passed,
            an instance of :class:`~google.auth.transport.requests.Request`
            is created.
        default_host (Optional[str]): A host like "pubsub.googleapis.com".
            This is used when a self-signed JWT is created from service
            account credentials.
    """

    def __init__(
        self,
        credentials,
        refresh_status_codes=transport.DEFAULT_REFRESH_STATUS_CODES,
        max_refresh_attempts=transport.DEFAULT_MAX_REFRESH_ATTEMPTS,
        refresh_timeout=None,
        auth_request=None,
        default_host=None,
    ):
        super(AuthorizedSession, self).__init__()
        self.credentials = credentials
        self._refresh_status_codes = refresh_status_codes
        self._max_refresh_attempts = max_refresh_attempts
        self._refresh_timeout = refresh_timeout
        self._is_mtls = False
        self._default_host = default_host

        if auth_request is None:
            self._auth_request_session = requests.Session()

            # Using an adapter to make HTTP requests robust to network errors.
            # This adapter retrys HTTP requests when network errors occur
            # and the requests seems safely retryable.
            retry_adapter = requests.adapters.HTTPAdapter(max_retries=3)
            self._auth_request_session.mount("https://", retry_adapter)

            # Do not pass `self` as the session here, as it can lead to
            # infinite recursion.
            auth_request = Request(self._auth_request_session)
        else:
            self._auth_request_session = None

        # Request instance used by internal methods (for example,
        # credentials.refresh).
        self._auth_request = auth_request

        # https://google.aip.dev/auth/4111
        # Attempt to use self-signed JWTs when a service account is used.
        if isinstance(self.credentials, service_account.Credentials):
            self.credentials._create_self_signed_jwt(
                "https://{}/".format(self._default_host) if self._default_host else None
            )

    def configure_mtls_channel(self, client_cert_callback=None):
        """Configure the client certificate and key for SSL connection.

        The function does nothing unless `GOOGLE_API_USE_CLIENT_CERTIFICATE` is
        explicitly set to `true`. In this case if client certificate and key are
        successfully obtained (from the given client_cert_callback or from application
        default SSL credentials), a :class:`_MutualTlsAdapter` instance will be mounted
        to "https://" prefix.

        Args:
            client_cert_callback (Optional[Callable[[], (bytes, bytes)]]):
                The optional callback returns the client certificate and private
                key bytes both in PEM format.
                If the callback is None, application default SSL credentials
                will be used.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS channel
                creation failed for any reason.
        """
        use_client_cert = os.getenv(
            environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE, "false"
        )
        if use_client_cert != "true":
            self._is_mtls = False
            return

        try:
            import OpenSSL
        except ImportError as caught_exc:
            new_exc = exceptions.MutualTLSChannelError(caught_exc)
            six.raise_from(new_exc, caught_exc)

        try:
            (
                self._is_mtls,
                cert,
                key,
            ) = google.auth.transport._mtls_helper.get_client_cert_and_key(
                client_cert_callback
            )

            if self._is_mtls:
                mtls_adapter = _MutualTlsAdapter(cert, key)
                self.mount("https://", mtls_adapter)
        except (
            exceptions.ClientCertError,
            ImportError,
            OpenSSL.crypto.Error,
        ) as caught_exc:
            new_exc = exceptions.MutualTLSChannelError(caught_exc)
            six.raise_from(new_exc, caught_exc)

    def request(
        self,
        method,
        url,
        data=None,
        headers=None,
        max_allowed_time=None,
        timeout=_DEFAULT_TIMEOUT,
        **kwargs
    ):
        """Implementation of Requests' request.

        Args:
            timeout (Optional[Union[float, Tuple[float, float]]]):
                The amount of time in seconds to wait for the server response
                with each individual request. Can also be passed as a tuple
                ``(connect_timeout, read_timeout)``. See :meth:`requests.Session.request`
                documentation for details.
            max_allowed_time (Optional[float]):
                If the method runs longer than this, a ``Timeout`` exception is
                automatically raised. Unlike the ``timeout`` parameter, this
                value applies to the total method execution time, even if
                multiple requests are made under the hood.

                Mind that it is not guaranteed that the timeout error is raised
                at ``max_allowed_time``. It might take longer, for example, if
                an underlying request takes a lot of time, but the request
                itself does not timeout, e.g. if a large file is being
                transmitted. The timout error will be raised after such
                request completes.
        """
        # pylint: disable=arguments-differ
        # Requests has a ton of arguments to request, but only two
        # (method, url) are required. We pass through all of the other
        # arguments to super, so no need to exhaustively list them here.

        # Use a kwarg for this instead of an attribute to maintain
        # thread-safety.
        _credential_refresh_attempt = kwargs.pop("_credential_refresh_attempt", 0)

        # Make a copy of the headers. They will be modified by the credentials
        # and we want to pass the original headers if we recurse.
        request_headers = headers.copy() if headers is not None else {}

        # Do not apply the timeout unconditionally in order to not override the
        # _auth_request's default timeout.
        auth_request = (
            self._auth_request
            if timeout is None
            else functools.partial(self._auth_request, timeout=timeout)
        )

        remaining_time = max_allowed_time

        with TimeoutGuard(remaining_time) as guard:
            self.credentials.before_request(auth_request, method, url, request_headers)
        remaining_time = guard.remaining_timeout

        with TimeoutGuard(remaining_time) as guard:
            response = super(AuthorizedSession, self).request(
                method,
                url,
                data=data,
                headers=request_headers,
                timeout=timeout,
                **kwargs
            )
        remaining_time = guard.remaining_timeout

        # If the response indicated that the credentials needed to be
        # refreshed, then refresh the credentials and re-attempt the
        # request.
        # A stored token may expire between the time it is retrieved and
        # the time the request is made, so we may need to try twice.
        if (
            response.status_code in self._refresh_status_codes
            and _credential_refresh_attempt < self._max_refresh_attempts
        ):

            _LOGGER.info(
                "Refreshing credentials due to a %s response. Attempt %s/%s.",
                response.status_code,
                _credential_refresh_attempt + 1,
                self._max_refresh_attempts,
            )

            # Do not apply the timeout unconditionally in order to not override the
            # _auth_request's default timeout.
            auth_request = (
                self._auth_request
                if timeout is None
                else functools.partial(self._auth_request, timeout=timeout)
            )

            with TimeoutGuard(remaining_time) as guard:
                self.credentials.refresh(auth_request)
            remaining_time = guard.remaining_timeout

            # Recurse. Pass in the original headers, not our modified set, but
            # do pass the adjusted max allowed time (i.e. the remaining total time).
            return self.request(
                method,
                url,
                data=data,
                headers=headers,
                max_allowed_time=remaining_time,
                timeout=timeout,
                _credential_refresh_attempt=_credential_refresh_attempt + 1,
                **kwargs
            )

        return response

    @property
    def is_mtls(self):
        """Indicates if the created SSL channel is mutual TLS."""
        return self._is_mtls

    def close(self):
        if self._auth_request_session is not None:
            self._auth_request_session.close()
        super(AuthorizedSession, self).close()
