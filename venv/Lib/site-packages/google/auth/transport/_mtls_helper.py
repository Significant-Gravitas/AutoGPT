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

"""Helper functions for getting mTLS cert and key."""

import json
import logging
from os import path
import re
import subprocess

import six

from google.auth import exceptions

CONTEXT_AWARE_METADATA_PATH = "~/.secureConnect/context_aware_metadata.json"
_CERT_PROVIDER_COMMAND = "cert_provider_command"
_CERT_REGEX = re.compile(
    b"-----BEGIN CERTIFICATE-----.+-----END CERTIFICATE-----\r?\n?", re.DOTALL
)

# support various format of key files, e.g.
# "-----BEGIN PRIVATE KEY-----...",
# "-----BEGIN EC PRIVATE KEY-----...",
# "-----BEGIN RSA PRIVATE KEY-----..."
# "-----BEGIN ENCRYPTED PRIVATE KEY-----"
_KEY_REGEX = re.compile(
    b"-----BEGIN [A-Z ]*PRIVATE KEY-----.+-----END [A-Z ]*PRIVATE KEY-----\r?\n?",
    re.DOTALL,
)

_LOGGER = logging.getLogger(__name__)


_PASSPHRASE_REGEX = re.compile(
    b"-----BEGIN PASSPHRASE-----(.+)-----END PASSPHRASE-----", re.DOTALL
)


def _check_dca_metadata_path(metadata_path):
    """Checks for context aware metadata. If it exists, returns the absolute path;
    otherwise returns None.

    Args:
        metadata_path (str): context aware metadata path.

    Returns:
        str: absolute path if exists and None otherwise.
    """
    metadata_path = path.expanduser(metadata_path)
    if not path.exists(metadata_path):
        _LOGGER.debug("%s is not found, skip client SSL authentication.", metadata_path)
        return None
    return metadata_path


def _read_dca_metadata_file(metadata_path):
    """Loads context aware metadata from the given path.

    Args:
        metadata_path (str): context aware metadata path.

    Returns:
        Dict[str, str]: The metadata.

    Raises:
        google.auth.exceptions.ClientCertError: If failed to parse metadata as JSON.
    """
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except ValueError as caught_exc:
        new_exc = exceptions.ClientCertError(caught_exc)
        six.raise_from(new_exc, caught_exc)

    return metadata


def _run_cert_provider_command(command, expect_encrypted_key=False):
    """Run the provided command, and return client side mTLS cert, key and
    passphrase.

    Args:
        command (List[str]): cert provider command.
        expect_encrypted_key (bool): If encrypted private key is expected.

    Returns:
        Tuple[bytes, bytes, bytes]: client certificate bytes in PEM format, key
            bytes in PEM format and passphrase bytes.

    Raises:
        google.auth.exceptions.ClientCertError: if problems occurs when running
            the cert provider command or generating cert, key and passphrase.
    """
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
    except OSError as caught_exc:
        new_exc = exceptions.ClientCertError(caught_exc)
        six.raise_from(new_exc, caught_exc)

    # Check cert provider command execution error.
    if process.returncode != 0:
        raise exceptions.ClientCertError(
            "Cert provider command returns non-zero status code %s" % process.returncode
        )

    # Extract certificate (chain), key and passphrase.
    cert_match = re.findall(_CERT_REGEX, stdout)
    if len(cert_match) != 1:
        raise exceptions.ClientCertError("Client SSL certificate is missing or invalid")
    key_match = re.findall(_KEY_REGEX, stdout)
    if len(key_match) != 1:
        raise exceptions.ClientCertError("Client SSL key is missing or invalid")
    passphrase_match = re.findall(_PASSPHRASE_REGEX, stdout)

    if expect_encrypted_key:
        if len(passphrase_match) != 1:
            raise exceptions.ClientCertError("Passphrase is missing or invalid")
        if b"ENCRYPTED" not in key_match[0]:
            raise exceptions.ClientCertError("Encrypted private key is expected")
        return cert_match[0], key_match[0], passphrase_match[0].strip()

    if b"ENCRYPTED" in key_match[0]:
        raise exceptions.ClientCertError("Encrypted private key is not expected")
    if len(passphrase_match) > 0:
        raise exceptions.ClientCertError("Passphrase is not expected")
    return cert_match[0], key_match[0], None


def get_client_ssl_credentials(
    generate_encrypted_key=False,
    context_aware_metadata_path=CONTEXT_AWARE_METADATA_PATH,
):
    """Returns the client side certificate, private key and passphrase.

    Args:
        generate_encrypted_key (bool): If set to True, encrypted private key
            and passphrase will be generated; otherwise, unencrypted private key
            will be generated and passphrase will be None.
        context_aware_metadata_path (str): The context_aware_metadata.json file path.

    Returns:
        Tuple[bool, bytes, bytes, bytes]:
            A boolean indicating if cert, key and passphrase are obtained, the
            cert bytes and key bytes both in PEM format, and passphrase bytes.

    Raises:
        google.auth.exceptions.ClientCertError: if problems occurs when getting
            the cert, key and passphrase.
    """
    metadata_path = _check_dca_metadata_path(context_aware_metadata_path)

    if metadata_path:
        metadata_json = _read_dca_metadata_file(metadata_path)

        if _CERT_PROVIDER_COMMAND not in metadata_json:
            raise exceptions.ClientCertError("Cert provider command is not found")

        command = metadata_json[_CERT_PROVIDER_COMMAND]

        if generate_encrypted_key and "--with_passphrase" not in command:
            command.append("--with_passphrase")

        # Execute the command.
        cert, key, passphrase = _run_cert_provider_command(
            command, expect_encrypted_key=generate_encrypted_key
        )
        return True, cert, key, passphrase

    return False, None, None, None


def get_client_cert_and_key(client_cert_callback=None):
    """Returns the client side certificate and private key. The function first
    tries to get certificate and key from client_cert_callback; if the callback
    is None or doesn't provide certificate and key, the function tries application
    default SSL credentials.

    Args:
        client_cert_callback (Optional[Callable[[], (bytes, bytes)]]): An
            optional callback which returns client certificate bytes and private
            key bytes both in PEM format.

    Returns:
        Tuple[bool, bytes, bytes]:
            A boolean indicating if cert and key are obtained, the cert bytes
            and key bytes both in PEM format.

    Raises:
        google.auth.exceptions.ClientCertError: if problems occurs when getting
            the cert and key.
    """
    if client_cert_callback:
        cert, key = client_cert_callback()
        return True, cert, key

    has_cert, cert, key, _ = get_client_ssl_credentials(generate_encrypted_key=False)
    return has_cert, cert, key


def decrypt_private_key(key, passphrase):
    """A helper function to decrypt the private key with the given passphrase.
    google-auth library doesn't support passphrase protected private key for
    mutual TLS channel. This helper function can be used to decrypt the
    passphrase protected private key in order to estalish mutual TLS channel.

    For example, if you have a function which produces client cert, passphrase
    protected private key and passphrase, you can convert it to a client cert
    callback function accepted by google-auth::

        from google.auth.transport import _mtls_helper

        def your_client_cert_function():
            return cert, encrypted_key, passphrase

        # callback accepted by google-auth for mutual TLS channel.
        def client_cert_callback():
            cert, encrypted_key, passphrase = your_client_cert_function()
            decrypted_key = _mtls_helper.decrypt_private_key(encrypted_key,
                passphrase)
            return cert, decrypted_key

    Args:
        key (bytes): The private key bytes in PEM format.
        passphrase (bytes): The passphrase bytes.

    Returns:
        bytes: The decrypted private key in PEM format.

    Raises:
        ImportError: If pyOpenSSL is not installed.
        OpenSSL.crypto.Error: If there is any problem decrypting the private key.
    """
    from OpenSSL import crypto

    # First convert encrypted_key_bytes to PKey object
    pkey = crypto.load_privatekey(crypto.FILETYPE_PEM, key, passphrase=passphrase)

    # Then dump the decrypted key bytes
    return crypto.dump_privatekey(crypto.FILETYPE_PEM, pkey)
