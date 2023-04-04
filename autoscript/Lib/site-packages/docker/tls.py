import os
import ssl

from . import errors
from .transport import SSLHTTPAdapter


class TLSConfig:
    """
    TLS configuration.

    Args:
        client_cert (tuple of str): Path to client cert, path to client key.
        ca_cert (str): Path to CA cert file.
        verify (bool or str): This can be a bool or a path to a CA cert
            file to verify against. If ``True``, verify using ca_cert;
            if ``False`` or not specified, do not verify.
        ssl_version (int): A valid `SSL version`_.
        assert_hostname (bool): Verify the hostname of the server.

    .. _`SSL version`:
        https://docs.python.org/3.5/library/ssl.html#ssl.PROTOCOL_TLSv1
    """
    cert = None
    ca_cert = None
    verify = None
    ssl_version = None

    def __init__(self, client_cert=None, ca_cert=None, verify=None,
                 ssl_version=None, assert_hostname=None,
                 assert_fingerprint=None):
        # Argument compatibility/mapping with
        # https://docs.docker.com/engine/articles/https/
        # This diverges from the Docker CLI in that users can specify 'tls'
        # here, but also disable any public/default CA pool verification by
        # leaving verify=False

        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint

        # If the user provides an SSL version, we should use their preference
        if ssl_version:
            self.ssl_version = ssl_version
        else:
            self.ssl_version = ssl.PROTOCOL_TLS_CLIENT

        # "client_cert" must have both or neither cert/key files. In
        # either case, Alert the user when both are expected, but any are
        # missing.

        if client_cert:
            try:
                tls_cert, tls_key = client_cert
            except ValueError:
                raise errors.TLSParameterError(
                    'client_cert must be a tuple of'
                    ' (client certificate, key file)'
                )

            if not (tls_cert and tls_key) or (not os.path.isfile(tls_cert) or
                                              not os.path.isfile(tls_key)):
                raise errors.TLSParameterError(
                    'Path to a certificate and key files must be provided'
                    ' through the client_cert param'
                )
            self.cert = (tls_cert, tls_key)

        # If verify is set, make sure the cert exists
        self.verify = verify
        self.ca_cert = ca_cert
        if self.verify and self.ca_cert and not os.path.isfile(self.ca_cert):
            raise errors.TLSParameterError(
                'Invalid CA certificate provided for `ca_cert`.'
            )

    def configure_client(self, client):
        """
        Configure a client with these TLS options.
        """
        client.ssl_version = self.ssl_version

        if self.verify and self.ca_cert:
            client.verify = self.ca_cert
        else:
            client.verify = self.verify

        if self.cert:
            client.cert = self.cert

        client.mount('https://', SSLHTTPAdapter(
            ssl_version=self.ssl_version,
            assert_hostname=self.assert_hostname,
            assert_fingerprint=self.assert_fingerprint,
        ))
