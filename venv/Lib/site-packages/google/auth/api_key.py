# Copyright 2022 Google LLC
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

"""Google API key support.
This module provides authentication using the `API key`_.
.. _API key:
    https://cloud.google.com/docs/authentication/api-keys/
"""

from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions


class Credentials(credentials.Credentials):
    """API key credentials.
    These credentials use API key to provide authorization to applications.
    """

    def __init__(self, token):
        """
        Args:
            token (str): API key string
        Raises:
            ValueError: If the provided API key is not a non-empty string.
        """
        super(Credentials, self).__init__()
        if not token:
            raise exceptions.InvalidValue("Token must be a non-empty API key string")
        self.token = token

    @property
    def expired(self):
        return False

    @property
    def valid(self):
        return True

    @_helpers.copy_docstring(credentials.Credentials)
    def refresh(self, request):
        return

    def apply(self, headers, token=None):
        """Apply the API key token to the x-goog-api-key header.
        Args:
            headers (Mapping): The HTTP request headers.
            token (Optional[str]): If specified, overrides the current access
                token.
        """
        headers["x-goog-api-key"] = token or self.token

    def before_request(self, request, method, url, headers):
        """Performs credential-specific before request logic.
        Refreshes the credentials if necessary, then calls :meth:`apply` to
        apply the token to the x-goog-api-key header.
        Args:
            request (google.auth.transport.Request): The object used to make
                HTTP requests.
            method (str): The request's HTTP method or the RPC method being
                invoked.
            url (str): The request's URI or the RPC service's URI.
            headers (Mapping): The request's headers.
        """
        self.apply(headers)
