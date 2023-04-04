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

"""OAuth 2.0 Token Exchange Spec.

This module defines a token exchange utility based on the `OAuth 2.0 Token
Exchange`_ spec. This will be mainly used to exchange external credentials
for GCP access tokens in workload identity pools to access Google APIs.

The implementation will support various types of client authentication as
allowed in the spec.

A deviation on the spec will be for additional Google specific options that
cannot be easily mapped to parameters defined in the RFC.

The returned dictionary response will be based on the `rfc8693 section 2.2.1`_
spec JSON response.

.. _OAuth 2.0 Token Exchange: https://tools.ietf.org/html/rfc8693
.. _rfc8693 section 2.2.1: https://tools.ietf.org/html/rfc8693#section-2.2.1
"""

import json

from six.moves import http_client
from six.moves import urllib

from google.oauth2 import utils


_URLENCODED_HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


class Client(utils.OAuthClientAuthHandler):
    """Implements the OAuth 2.0 token exchange spec based on
    https://tools.ietf.org/html/rfc8693.
    """

    def __init__(self, token_exchange_endpoint, client_authentication=None):
        """Initializes an STS client instance.

        Args:
            token_exchange_endpoint (str): The token exchange endpoint.
            client_authentication (Optional(google.oauth2.oauth2_utils.ClientAuthentication)):
                The optional OAuth client authentication credentials if available.
        """
        super(Client, self).__init__(client_authentication)
        self._token_exchange_endpoint = token_exchange_endpoint

    def _make_request(self, request, headers, request_body):
        # Initialize request headers.
        request_headers = _URLENCODED_HEADERS.copy()

        # Inject additional headers.
        if headers:
            for k, v in dict(headers).items():
                request_headers[k] = v

        # Apply OAuth client authentication.
        self.apply_client_authentication_options(request_headers, request_body)

        # Execute request.
        response = request(
            url=self._token_exchange_endpoint,
            method="POST",
            headers=request_headers,
            body=urllib.parse.urlencode(request_body).encode("utf-8"),
        )

        response_body = (
            response.data.decode("utf-8")
            if hasattr(response.data, "decode")
            else response.data
        )

        # If non-200 response received, translate to OAuthError exception.
        if response.status != http_client.OK:
            utils.handle_error_response(response_body)

        response_data = json.loads(response_body)

        # Return successful response.
        return response_data

    def exchange_token(
        self,
        request,
        grant_type,
        subject_token,
        subject_token_type,
        resource=None,
        audience=None,
        scopes=None,
        requested_token_type=None,
        actor_token=None,
        actor_token_type=None,
        additional_options=None,
        additional_headers=None,
    ):
        """Exchanges the provided token for another type of token based on the
        rfc8693 spec.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            grant_type (str): The OAuth 2.0 token exchange grant type.
            subject_token (str): The OAuth 2.0 token exchange subject token.
            subject_token_type (str): The OAuth 2.0 token exchange subject token type.
            resource (Optional[str]): The optional OAuth 2.0 token exchange resource field.
            audience (Optional[str]): The optional OAuth 2.0 token exchange audience field.
            scopes (Optional[Sequence[str]]): The optional list of scopes to use.
            requested_token_type (Optional[str]): The optional OAuth 2.0 token exchange requested
                token type.
            actor_token (Optional[str]): The optional OAuth 2.0 token exchange actor token.
            actor_token_type (Optional[str]): The optional OAuth 2.0 token exchange actor token type.
            additional_options (Optional[Mapping[str, str]]): The optional additional
                non-standard Google specific options.
            additional_headers (Optional[Mapping[str, str]]): The optional additional
                headers to pass to the token exchange endpoint.

        Returns:
            Mapping[str, str]: The token exchange JSON-decoded response data containing
                the requested token and its expiration time.

        Raises:
            google.auth.exceptions.OAuthError: If the token endpoint returned
                an error.
        """
        # Initialize request body.
        request_body = {
            "grant_type": grant_type,
            "resource": resource,
            "audience": audience,
            "scope": " ".join(scopes or []),
            "requested_token_type": requested_token_type,
            "subject_token": subject_token,
            "subject_token_type": subject_token_type,
            "actor_token": actor_token,
            "actor_token_type": actor_token_type,
            "options": None,
        }
        # Add additional non-standard options.
        if additional_options:
            request_body["options"] = urllib.parse.quote(json.dumps(additional_options))
        # Remove empty fields in request body.
        for k, v in dict(request_body).items():
            if v is None or v == "":
                del request_body[k]

        return self._make_request(request, additional_headers, request_body)

    def refresh_token(self, request, refresh_token):
        """Exchanges a refresh token for an access token based on the
        RFC6749 spec.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            subject_token (str): The OAuth 2.0 refresh token.
        """

        return self._make_request(
            request,
            None,
            {"grant_type": "refresh_token", "refresh_token": refresh_token},
        )
