# Copyright 2020 Google LLC
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

"""Utilites for mutual TLS."""

import six

from google.auth import exceptions
from google.auth.transport import _mtls_helper


def has_default_client_cert_source():
    """Check if default client SSL credentials exists on the device.

    Returns:
        bool: indicating if the default client cert source exists.
    """
    metadata_path = _mtls_helper._check_dca_metadata_path(
        _mtls_helper.CONTEXT_AWARE_METADATA_PATH
    )
    return metadata_path is not None


def default_client_cert_source():
    """Get a callback which returns the default client SSL credentials.

    Returns:
        Callable[[], [bytes, bytes]]: A callback which returns the default
            client certificate bytes and private key bytes, both in PEM format.

    Raises:
        google.auth.exceptions.DefaultClientCertSourceError: If the default
            client SSL credentials don't exist or are malformed.
    """
    if not has_default_client_cert_source():
        raise exceptions.MutualTLSChannelError(
            "Default client cert source doesn't exist"
        )

    def callback():
        try:
            _, cert_bytes, key_bytes = _mtls_helper.get_client_cert_and_key()
        except (OSError, RuntimeError, ValueError) as caught_exc:
            new_exc = exceptions.MutualTLSChannelError(caught_exc)
            six.raise_from(new_exc, caught_exc)

        return cert_bytes, key_bytes

    return callback


def default_client_encrypted_cert_source(cert_path, key_path):
    """Get a callback which returns the default encrpyted client SSL credentials.

    Args:
        cert_path (str): The cert file path. The default client certificate will
            be written to this file when the returned callback is called.
        key_path (str): The key file path. The default encrypted client key will
            be written to this file when the returned callback is called.

    Returns:
        Callable[[], [str, str, bytes]]: A callback which generates the default
            client certificate, encrpyted private key and passphrase. It writes
            the certificate and private key into the cert_path and key_path, and
            returns the cert_path, key_path and passphrase bytes.

    Raises:
        google.auth.exceptions.DefaultClientCertSourceError: If any problem
            occurs when loading or saving the client certificate and key.
    """
    if not has_default_client_cert_source():
        raise exceptions.MutualTLSChannelError(
            "Default client encrypted cert source doesn't exist"
        )

    def callback():
        try:
            (
                _,
                cert_bytes,
                key_bytes,
                passphrase_bytes,
            ) = _mtls_helper.get_client_ssl_credentials(generate_encrypted_key=True)
            with open(cert_path, "wb") as cert_file:
                cert_file.write(cert_bytes)
            with open(key_path, "wb") as key_file:
                key_file.write(key_bytes)
        except (exceptions.ClientCertError, OSError) as caught_exc:
            new_exc = exceptions.MutualTLSChannelError(caught_exc)
            six.raise_from(new_exc, caught_exc)

        return cert_path, key_path, passphrase_bytes

    return callback
