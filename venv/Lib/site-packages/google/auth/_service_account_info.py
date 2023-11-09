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

"""Helper functions for loading data from a Google service account file."""

import io
import json

import six

from google.auth import crypt
from google.auth import exceptions


def from_dict(data, require=None, use_rsa_signer=True):
    """Validates a dictionary containing Google service account data.

    Creates and returns a :class:`google.auth.crypt.Signer` instance from the
    private key specified in the data.

    Args:
        data (Mapping[str, str]): The service account data
        require (Sequence[str]): List of keys required to be present in the
            info.
        use_rsa_signer (Optional[bool]): Whether to use RSA signer or EC signer.
            We use RSA signer by default.

    Returns:
        google.auth.crypt.Signer: A signer created from the private key in the
            service account file.

    Raises:
        MalformedError: if the data was in the wrong format, or if one of the
            required keys is missing.
    """
    keys_needed = set(require if require is not None else [])

    missing = keys_needed.difference(six.iterkeys(data))

    if missing:
        raise exceptions.MalformedError(
            "Service account info was not in the expected format, missing "
            "fields {}.".format(", ".join(missing))
        )

    # Create a signer.
    if use_rsa_signer:
        signer = crypt.RSASigner.from_service_account_info(data)
    else:
        signer = crypt.ES256Signer.from_service_account_info(data)

    return signer


def from_filename(filename, require=None, use_rsa_signer=True):
    """Reads a Google service account JSON file and returns its parsed info.

    Args:
        filename (str): The path to the service account .json file.
        require (Sequence[str]): List of keys required to be present in the
            info.
        use_rsa_signer (Optional[bool]): Whether to use RSA signer or EC signer.
            We use RSA signer by default.

    Returns:
        Tuple[ Mapping[str, str], google.auth.crypt.Signer ]: The verified
            info and a signer instance.
    """
    with io.open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data, from_dict(data, require=require, use_rsa_signer=use_rsa_signer)
