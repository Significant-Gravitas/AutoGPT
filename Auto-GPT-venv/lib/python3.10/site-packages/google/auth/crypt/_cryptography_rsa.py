# Copyright 2017 Google LLC
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

"""RSA verifier and signer that use the ``cryptography`` library.

This is a much faster implementation than the default (in
``google.auth.crypt._python_rsa``), which depends on the pure-Python
``rsa`` library.
"""

import cryptography.exceptions
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
import cryptography.x509

from google.auth import _helpers
from google.auth.crypt import base

_CERTIFICATE_MARKER = b"-----BEGIN CERTIFICATE-----"
_BACKEND = backends.default_backend()
_PADDING = padding.PKCS1v15()
_SHA256 = hashes.SHA256()


class RSAVerifier(base.Verifier):
    """Verifies RSA cryptographic signatures using public keys.

    Args:
        public_key (
                cryptography.hazmat.primitives.asymmetric.rsa.RSAPublicKey):
            The public key used to verify signatures.
    """

    def __init__(self, public_key):
        self._pubkey = public_key

    @_helpers.copy_docstring(base.Verifier)
    def verify(self, message, signature):
        message = _helpers.to_bytes(message)
        try:
            self._pubkey.verify(signature, message, _PADDING, _SHA256)
            return True
        except (ValueError, cryptography.exceptions.InvalidSignature):
            return False

    @classmethod
    def from_string(cls, public_key):
        """Construct an Verifier instance from a public key or public
        certificate string.

        Args:
            public_key (Union[str, bytes]): The public key in PEM format or the
                x509 public key certificate.

        Returns:
            Verifier: The constructed verifier.

        Raises:
            ValueError: If the public key can't be parsed.
        """
        public_key_data = _helpers.to_bytes(public_key)

        if _CERTIFICATE_MARKER in public_key_data:
            cert = cryptography.x509.load_pem_x509_certificate(
                public_key_data, _BACKEND
            )
            pubkey = cert.public_key()

        else:
            pubkey = serialization.load_pem_public_key(public_key_data, _BACKEND)

        return cls(pubkey)


class RSASigner(base.Signer, base.FromServiceAccountMixin):
    """Signs messages with an RSA private key.

    Args:
        private_key (
                cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey):
            The private key to sign with.
        key_id (str): Optional key ID used to identify this private key. This
            can be useful to associate the private key with its associated
            public key or certificate.
    """

    def __init__(self, private_key, key_id=None):
        self._key = private_key
        self._key_id = key_id

    @property  # type: ignore
    @_helpers.copy_docstring(base.Signer)
    def key_id(self):
        return self._key_id

    @_helpers.copy_docstring(base.Signer)
    def sign(self, message):
        message = _helpers.to_bytes(message)
        return self._key.sign(message, _PADDING, _SHA256)

    @classmethod
    def from_string(cls, key, key_id=None):
        """Construct a RSASigner from a private key in PEM format.

        Args:
            key (Union[bytes, str]): Private key in PEM format.
            key_id (str): An optional key id used to identify the private key.

        Returns:
            google.auth.crypt._cryptography_rsa.RSASigner: The
            constructed signer.

        Raises:
            ValueError: If ``key`` is not ``bytes`` or ``str`` (unicode).
            UnicodeDecodeError: If ``key`` is ``bytes`` but cannot be decoded
                into a UTF-8 ``str``.
            ValueError: If ``cryptography`` "Could not deserialize key data."
        """
        key = _helpers.to_bytes(key)
        private_key = serialization.load_pem_private_key(
            key, password=None, backend=_BACKEND
        )
        return cls(private_key, key_id=key_id)
