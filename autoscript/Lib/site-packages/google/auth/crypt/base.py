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

"""Base classes for cryptographic signers and verifiers."""

import abc
import io
import json

import six

from google.auth import exceptions

_JSON_FILE_PRIVATE_KEY = "private_key"
_JSON_FILE_PRIVATE_KEY_ID = "private_key_id"


@six.add_metaclass(abc.ABCMeta)
class Verifier(object):
    """Abstract base class for crytographic signature verifiers."""

    @abc.abstractmethod
    def verify(self, message, signature):
        """Verifies a message against a cryptographic signature.

        Args:
            message (Union[str, bytes]): The message to verify.
            signature (Union[str, bytes]): The cryptography signature to check.

        Returns:
            bool: True if message was signed by the private key associated
            with the public key that this object was constructed with.
        """
        # pylint: disable=missing-raises-doc,redundant-returns-doc
        # (pylint doesn't recognize that this is abstract)
        raise NotImplementedError("Verify must be implemented")


@six.add_metaclass(abc.ABCMeta)
class Signer(object):
    """Abstract base class for cryptographic signers."""

    @abc.abstractproperty
    def key_id(self):
        """Optional[str]: The key ID used to identify this private key."""
        raise NotImplementedError("Key id must be implemented")

    @abc.abstractmethod
    def sign(self, message):
        """Signs a message.

        Args:
            message (Union[str, bytes]): The message to be signed.

        Returns:
            bytes: The signature of the message.
        """
        # pylint: disable=missing-raises-doc,redundant-returns-doc
        # (pylint doesn't recognize that this is abstract)
        raise NotImplementedError("Sign must be implemented")


@six.add_metaclass(abc.ABCMeta)
class FromServiceAccountMixin(object):
    """Mix-in to enable factory constructors for a Signer."""

    @abc.abstractmethod
    def from_string(cls, key, key_id=None):
        """Construct an Signer instance from a private key string.

        Args:
            key (str): Private key as a string.
            key_id (str): An optional key id used to identify the private key.

        Returns:
            google.auth.crypt.Signer: The constructed signer.

        Raises:
            ValueError: If the key cannot be parsed.
        """
        raise NotImplementedError("from_string must be implemented")

    @classmethod
    def from_service_account_info(cls, info):
        """Creates a Signer instance instance from a dictionary containing
        service account info in Google format.

        Args:
            info (Mapping[str, str]): The service account info in Google
                format.

        Returns:
            google.auth.crypt.Signer: The constructed signer.

        Raises:
            ValueError: If the info is not in the expected format.
        """
        if _JSON_FILE_PRIVATE_KEY not in info:
            raise exceptions.MalformedError(
                "The private_key field was not found in the service account " "info."
            )

        return cls.from_string(
            info[_JSON_FILE_PRIVATE_KEY], info.get(_JSON_FILE_PRIVATE_KEY_ID)
        )

    @classmethod
    def from_service_account_file(cls, filename):
        """Creates a Signer instance from a service account .json file
        in Google format.

        Args:
            filename (str): The path to the service account .json file.

        Returns:
            google.auth.crypt.Signer: The constructed signer.
        """
        with io.open(filename, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        return cls.from_service_account_info(data)
