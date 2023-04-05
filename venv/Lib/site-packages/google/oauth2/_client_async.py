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

"""OAuth 2.0 async client.

This is a client for interacting with an OAuth 2.0 authorization server's
token endpoint.

For more information about the token endpoint, see
`Section 3.1 of rfc6749`_

.. _Section 3.1 of rfc6749: https://tools.ietf.org/html/rfc6749#section-3.2
"""

import datetime
import json

import six
from six.moves import http_client
from six.moves import urllib

from google.auth import _exponential_backoff
from google.auth import exceptions
from google.auth import jwt
from google.oauth2 import _client as client


async def _token_endpoint_request_no_throw(
    request, token_uri, body, access_token=None, use_json=False, can_retry=True
):
    """Makes a request to the OAuth 2.0 authorization server's token endpoint.
    This function doesn't throw on response errors.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        token_uri (str): The OAuth 2.0 authorizations server's token endpoint
            URI.
        body (Mapping[str, str]): The parameters to send in the request body.
        access_token (Optional(str)): The access token needed to make the request.
        use_json (Optional(bool)): Use urlencoded format or json format for the
            content type. The default value is False.
        can_retry (bool): Enable or disable request retry behavior.

    Returns:
        Tuple(bool, Mapping[str, str], Optional[bool]): A boolean indicating
          if the request is successful, a mapping for the JSON-decoded response
          data and in the case of an error a boolean indicating if the error
          is retryable.
    """
    if use_json:
        headers = {"Content-Type": client._JSON_CONTENT_TYPE}
        body = json.dumps(body).encode("utf-8")
    else:
        headers = {"Content-Type": client._URLENCODED_CONTENT_TYPE}
        body = urllib.parse.urlencode(body).encode("utf-8")

    if access_token:
        headers["Authorization"] = "Bearer {}".format(access_token)

    async def _perform_request():
        response = await request(
            method="POST", url=token_uri, headers=headers, body=body
        )

        # Using data.read() resulted in zlib decompression errors. This may require future investigation.
        response_body1 = await response.content()

        response_body = (
            response_body1.decode("utf-8")
            if hasattr(response_body1, "decode")
            else response_body1
        )

        try:
            response_data = json.loads(response_body)
        except ValueError:
            response_data = response_body

        if response.status == http_client.OK:
            return True, response_data, None

        retryable_error = client._can_retry(
            status_code=response.status, response_data=response_data
        )

        return False, response_data, retryable_error

    request_succeeded, response_data, retryable_error = await _perform_request()

    if request_succeeded or not retryable_error or not can_retry:
        return request_succeeded, response_data, retryable_error

    retries = _exponential_backoff.ExponentialBackoff()
    for _ in retries:
        request_succeeded, response_data, retryable_error = await _perform_request()
        if request_succeeded or not retryable_error:
            return request_succeeded, response_data, retryable_error

    return False, response_data, retryable_error


async def _token_endpoint_request(
    request, token_uri, body, access_token=None, use_json=False, can_retry=True
):
    """Makes a request to the OAuth 2.0 authorization server's token endpoint.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        token_uri (str): The OAuth 2.0 authorizations server's token endpoint
            URI.
        body (Mapping[str, str]): The parameters to send in the request body.
        access_token (Optional(str)): The access token needed to make the request.
        use_json (Optional(bool)): Use urlencoded format or json format for the
            content type. The default value is False.
        can_retry (bool): Enable or disable request retry behavior.

    Returns:
        Mapping[str, str]: The JSON-decoded response data.

    Raises:
        google.auth.exceptions.RefreshError: If the token endpoint returned
            an error.
    """

    response_status_ok, response_data, retryable_error = await _token_endpoint_request_no_throw(
        request,
        token_uri,
        body,
        access_token=access_token,
        use_json=use_json,
        can_retry=can_retry,
    )
    if not response_status_ok:
        client._handle_error_response(response_data, retryable_error)
    return response_data


async def jwt_grant(request, token_uri, assertion, can_retry=True):
    """Implements the JWT Profile for OAuth 2.0 Authorization Grants.

    For more details, see `rfc7523 section 4`_.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        token_uri (str): The OAuth 2.0 authorizations server's token endpoint
            URI.
        assertion (str): The OAuth 2.0 assertion.
        can_retry (bool): Enable or disable request retry behavior.

    Returns:
        Tuple[str, Optional[datetime], Mapping[str, str]]: The access token,
            expiration, and additional data returned by the token endpoint.

    Raises:
        google.auth.exceptions.RefreshError: If the token endpoint returned
            an error.

    .. _rfc7523 section 4: https://tools.ietf.org/html/rfc7523#section-4
    """
    body = {"assertion": assertion, "grant_type": client._JWT_GRANT_TYPE}

    response_data = await _token_endpoint_request(
        request, token_uri, body, can_retry=can_retry
    )

    try:
        access_token = response_data["access_token"]
    except KeyError as caught_exc:
        new_exc = exceptions.RefreshError(
            "No access token in response.", response_data, retryable=False
        )
        six.raise_from(new_exc, caught_exc)

    expiry = client._parse_expiry(response_data)

    return access_token, expiry, response_data


async def id_token_jwt_grant(request, token_uri, assertion, can_retry=True):
    """Implements the JWT Profile for OAuth 2.0 Authorization Grants, but
    requests an OpenID Connect ID Token instead of an access token.

    This is a variant on the standard JWT Profile that is currently unique
    to Google. This was added for the benefit of authenticating to services
    that require ID Tokens instead of access tokens or JWT bearer tokens.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        token_uri (str): The OAuth 2.0 authorization server's token endpoint
            URI.
        assertion (str): JWT token signed by a service account. The token's
            payload must include a ``target_audience`` claim.
        can_retry (bool): Enable or disable request retry behavior.

    Returns:
        Tuple[str, Optional[datetime], Mapping[str, str]]:
            The (encoded) Open ID Connect ID Token, expiration, and additional
            data returned by the endpoint.

    Raises:
        google.auth.exceptions.RefreshError: If the token endpoint returned
            an error.
    """
    body = {"assertion": assertion, "grant_type": client._JWT_GRANT_TYPE}

    response_data = await _token_endpoint_request(
        request, token_uri, body, can_retry=can_retry
    )

    try:
        id_token = response_data["id_token"]
    except KeyError as caught_exc:
        new_exc = exceptions.RefreshError(
            "No ID token in response.", response_data, retryable=False
        )
        six.raise_from(new_exc, caught_exc)

    payload = jwt.decode(id_token, verify=False)
    expiry = datetime.datetime.utcfromtimestamp(payload["exp"])

    return id_token, expiry, response_data


async def refresh_grant(
    request,
    token_uri,
    refresh_token,
    client_id,
    client_secret,
    scopes=None,
    rapt_token=None,
    can_retry=True,
):
    """Implements the OAuth 2.0 refresh token grant.

    For more details, see `rfc678 section 6`_.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        token_uri (str): The OAuth 2.0 authorizations server's token endpoint
            URI.
        refresh_token (str): The refresh token to use to get a new access
            token.
        client_id (str): The OAuth 2.0 application's client ID.
        client_secret (str): The Oauth 2.0 appliaction's client secret.
        scopes (Optional(Sequence[str])): Scopes to request. If present, all
            scopes must be authorized for the refresh token. Useful if refresh
            token has a wild card scope (e.g.
            'https://www.googleapis.com/auth/any-api').
        rapt_token (Optional(str)): The reauth Proof Token.
        can_retry (bool): Enable or disable request retry behavior.

    Returns:
        Tuple[str, Optional[str], Optional[datetime], Mapping[str, str]]: The
            access token, new or current refresh token, expiration, and additional data
            returned by the token endpoint.

    Raises:
        google.auth.exceptions.RefreshError: If the token endpoint returned
            an error.

    .. _rfc6748 section 6: https://tools.ietf.org/html/rfc6749#section-6
    """
    body = {
        "grant_type": client._REFRESH_GRANT_TYPE,
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }
    if scopes:
        body["scope"] = " ".join(scopes)
    if rapt_token:
        body["rapt"] = rapt_token

    response_data = await _token_endpoint_request(
        request, token_uri, body, can_retry=can_retry
    )
    return client._handle_refresh_grant_response(response_data, refresh_token)
