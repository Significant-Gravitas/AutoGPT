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

"""Authorization support for gRPC."""

from __future__ import absolute_import

import logging
import os

import six

from google.auth import environment_vars
from google.auth import exceptions
from google.auth.transport import _mtls_helper
from google.oauth2 import service_account

try:
    import grpc  # type: ignore
except ImportError as caught_exc:  # pragma: NO COVER
    six.raise_from(
        ImportError(
            "gRPC is not installed, please install the grpcio package "
            "to use the gRPC transport."
        ),
        caught_exc,
    )

_LOGGER = logging.getLogger(__name__)


class AuthMetadataPlugin(grpc.AuthMetadataPlugin):
    """A `gRPC AuthMetadataPlugin`_ that inserts the credentials into each
    request.

    .. _gRPC AuthMetadataPlugin:
        http://www.grpc.io/grpc/python/grpc.html#grpc.AuthMetadataPlugin

    Args:
        credentials (google.auth.credentials.Credentials): The credentials to
            add to requests.
        request (google.auth.transport.Request): A HTTP transport request
            object used to refresh credentials as needed.
        default_host (Optional[str]): A host like "pubsub.googleapis.com".
            This is used when a self-signed JWT is created from service
            account credentials.
    """

    def __init__(self, credentials, request, default_host=None):
        # pylint: disable=no-value-for-parameter
        # pylint doesn't realize that the super method takes no arguments
        # because this class is the same name as the superclass.
        super(AuthMetadataPlugin, self).__init__()
        self._credentials = credentials
        self._request = request
        self._default_host = default_host

    def _get_authorization_headers(self, context):
        """Gets the authorization headers for a request.

        Returns:
            Sequence[Tuple[str, str]]: A list of request headers (key, value)
                to add to the request.
        """
        headers = {}

        # https://google.aip.dev/auth/4111
        # Attempt to use self-signed JWTs when a service account is used.
        # A default host must be explicitly provided since it cannot always
        # be determined from the context.service_url.
        if isinstance(self._credentials, service_account.Credentials):
            self._credentials._create_self_signed_jwt(
                "https://{}/".format(self._default_host) if self._default_host else None
            )

        self._credentials.before_request(
            self._request, context.method_name, context.service_url, headers
        )

        return list(six.iteritems(headers))

    def __call__(self, context, callback):
        """Passes authorization metadata into the given callback.

        Args:
            context (grpc.AuthMetadataContext): The RPC context.
            callback (grpc.AuthMetadataPluginCallback): The callback that will
                be invoked to pass in the authorization metadata.
        """
        callback(self._get_authorization_headers(context), None)


def secure_authorized_channel(
    credentials,
    request,
    target,
    ssl_credentials=None,
    client_cert_callback=None,
    **kwargs
):
    """Creates a secure authorized gRPC channel.

    This creates a channel with SSL and :class:`AuthMetadataPlugin`. This
    channel can be used to create a stub that can make authorized requests.
    Users can configure client certificate or rely on device certificates to
    establish a mutual TLS channel, if the `GOOGLE_API_USE_CLIENT_CERTIFICATE`
    variable is explicitly set to `true`.

    Example::

        import google.auth
        import google.auth.transport.grpc
        import google.auth.transport.requests
        from google.cloud.speech.v1 import cloud_speech_pb2

        # Get credentials.
        credentials, _ = google.auth.default()

        # Get an HTTP request function to refresh credentials.
        request = google.auth.transport.requests.Request()

        # Create a channel.
        channel = google.auth.transport.grpc.secure_authorized_channel(
            credentials, regular_endpoint, request,
            ssl_credentials=grpc.ssl_channel_credentials())

        # Use the channel to create a stub.
        cloud_speech.create_Speech_stub(channel)

    Usage:

    There are actually a couple of options to create a channel, depending on if
    you want to create a regular or mutual TLS channel.

    First let's list the endpoints (regular vs mutual TLS) to choose from::

        regular_endpoint = 'speech.googleapis.com:443'
        mtls_endpoint = 'speech.mtls.googleapis.com:443'

    Option 1: create a regular (non-mutual) TLS channel by explicitly setting
    the ssl_credentials::

        regular_ssl_credentials = grpc.ssl_channel_credentials()

        channel = google.auth.transport.grpc.secure_authorized_channel(
            credentials, regular_endpoint, request,
            ssl_credentials=regular_ssl_credentials)

    Option 2: create a mutual TLS channel by calling a callback which returns
    the client side certificate and the key (Note that
    `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable must be explicitly
    set to `true`)::

        def my_client_cert_callback():
            code_to_load_client_cert_and_key()
            if loaded:
                return (pem_cert_bytes, pem_key_bytes)
            raise MyClientCertFailureException()

        try:
            channel = google.auth.transport.grpc.secure_authorized_channel(
                credentials, mtls_endpoint, request,
                client_cert_callback=my_client_cert_callback)
        except MyClientCertFailureException:
            # handle the exception

    Option 3: use application default SSL credentials. It searches and uses
    the command in a context aware metadata file, which is available on devices
    with endpoint verification support (Note that
    `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable must be explicitly
    set to `true`).
    See https://cloud.google.com/endpoint-verification/docs/overview::

        try:
            default_ssl_credentials = SslCredentials()
        except:
            # Exception can be raised if the context aware metadata is malformed.
            # See :class:`SslCredentials` for the possible exceptions.

        # Choose the endpoint based on the SSL credentials type.
        if default_ssl_credentials.is_mtls:
            endpoint_to_use = mtls_endpoint
        else:
            endpoint_to_use = regular_endpoint
        channel = google.auth.transport.grpc.secure_authorized_channel(
            credentials, endpoint_to_use, request,
            ssl_credentials=default_ssl_credentials)

    Option 4: not setting ssl_credentials and client_cert_callback. For devices
    without endpoint verification support or `GOOGLE_API_USE_CLIENT_CERTIFICATE`
    environment variable is not `true`, a regular TLS channel is created;
    otherwise, a mutual TLS channel is created, however, the call should be
    wrapped in a try/except block in case of malformed context aware metadata.

    The following code uses regular_endpoint, it works the same no matter the
    created channle is regular or mutual TLS. Regular endpoint ignores client
    certificate and key::

        channel = google.auth.transport.grpc.secure_authorized_channel(
            credentials, regular_endpoint, request)

    The following code uses mtls_endpoint, if the created channle is regular,
    and API mtls_endpoint is confgured to require client SSL credentials, API
    calls using this channel will be rejected::

        channel = google.auth.transport.grpc.secure_authorized_channel(
            credentials, mtls_endpoint, request)

    Args:
        credentials (google.auth.credentials.Credentials): The credentials to
            add to requests.
        request (google.auth.transport.Request): A HTTP transport request
            object used to refresh credentials as needed. Even though gRPC
            is a separate transport, there's no way to refresh the credentials
            without using a standard http transport.
        target (str): The host and port of the service.
        ssl_credentials (grpc.ChannelCredentials): Optional SSL channel
            credentials. This can be used to specify different certificates.
            This argument is mutually exclusive with client_cert_callback;
            providing both will raise an exception.
            If ssl_credentials and client_cert_callback are None, application
            default SSL credentials are used if `GOOGLE_API_USE_CLIENT_CERTIFICATE`
            environment variable is explicitly set to `true`, otherwise one way TLS
            SSL credentials are used.
        client_cert_callback (Callable[[], (bytes, bytes)]): Optional
            callback function to obtain client certicate and key for mutual TLS
            connection. This argument is mutually exclusive with
            ssl_credentials; providing both will raise an exception.
            This argument does nothing unless `GOOGLE_API_USE_CLIENT_CERTIFICATE`
            environment variable is explicitly set to `true`.
        kwargs: Additional arguments to pass to :func:`grpc.secure_channel`.

    Returns:
        grpc.Channel: The created gRPC channel.

    Raises:
        google.auth.exceptions.MutualTLSChannelError: If mutual TLS channel
            creation failed for any reason.
    """
    # Create the metadata plugin for inserting the authorization header.
    metadata_plugin = AuthMetadataPlugin(credentials, request)

    # Create a set of grpc.CallCredentials using the metadata plugin.
    google_auth_credentials = grpc.metadata_call_credentials(metadata_plugin)

    if ssl_credentials and client_cert_callback:
        raise exceptions.MalformedError(
            "Received both ssl_credentials and client_cert_callback; "
            "these are mutually exclusive."
        )

    # If SSL credentials are not explicitly set, try client_cert_callback and ADC.
    if not ssl_credentials:
        use_client_cert = os.getenv(
            environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE, "false"
        )
        if use_client_cert == "true" and client_cert_callback:
            # Use the callback if provided.
            cert, key = client_cert_callback()
            ssl_credentials = grpc.ssl_channel_credentials(
                certificate_chain=cert, private_key=key
            )
        elif use_client_cert == "true":
            # Use application default SSL credentials.
            adc_ssl_credentils = SslCredentials()
            ssl_credentials = adc_ssl_credentils.ssl_credentials
        else:
            ssl_credentials = grpc.ssl_channel_credentials()

    # Combine the ssl credentials and the authorization credentials.
    composite_credentials = grpc.composite_channel_credentials(
        ssl_credentials, google_auth_credentials
    )

    return grpc.secure_channel(target, composite_credentials, **kwargs)


class SslCredentials:
    """Class for application default SSL credentials.

    The behavior is controlled by `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment
    variable whose default value is `false`. Client certificate will not be used
    unless the environment variable is explicitly set to `true`. See
    https://google.aip.dev/auth/4114

    If the environment variable is `true`, then for devices with endpoint verification
    support, a device certificate will be automatically loaded and mutual TLS will
    be established.
    See https://cloud.google.com/endpoint-verification/docs/overview.
    """

    def __init__(self):
        use_client_cert = os.getenv(
            environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE, "false"
        )
        if use_client_cert != "true":
            self._is_mtls = False
        else:
            # Load client SSL credentials.
            metadata_path = _mtls_helper._check_dca_metadata_path(
                _mtls_helper.CONTEXT_AWARE_METADATA_PATH
            )
            self._is_mtls = metadata_path is not None

    @property
    def ssl_credentials(self):
        """Get the created SSL channel credentials.

        For devices with endpoint verification support, if the device certificate
        loading has any problems, corresponding exceptions will be raised. For
        a device without endpoint verification support, no exceptions will be
        raised.

        Returns:
            grpc.ChannelCredentials: The created grpc channel credentials.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS channel
                creation failed for any reason.
        """
        if self._is_mtls:
            try:
                _, cert, key, _ = _mtls_helper.get_client_ssl_credentials()
                self._ssl_credentials = grpc.ssl_channel_credentials(
                    certificate_chain=cert, private_key=key
                )
            except exceptions.ClientCertError as caught_exc:
                new_exc = exceptions.MutualTLSChannelError(caught_exc)
                six.raise_from(new_exc, caught_exc)
        else:
            self._ssl_credentials = grpc.ssl_channel_credentials()

        return self._ssl_credentials

    @property
    def is_mtls(self):
        """Indicates if the created SSL channel credentials is mutual TLS."""
        return self._is_mtls
