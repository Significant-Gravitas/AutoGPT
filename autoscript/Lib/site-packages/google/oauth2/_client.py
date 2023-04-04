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

"""OAuth 2.0 client.

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
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport

_URLENCODED_CONTENT_TYPE = "application/x-www-form-urlencoded"
_JSON_CONTENT_TYPE = "application/json"
_JWT_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_REFRESH_GRANT_TYPE = "refresh_token"
_IAM_IDTOKEN_ENDPOINT = (
    "https://iamcredentials.googleapis.com/v1/"
    + "projects/-/serviceAccounts/{}:generateIdToken"
)


def _handle_error_response(response_data, retryable_error):
    """Translates an error response into an exception.

    Args:
        response_data (Mapping | str): The decoded response data.
        retryable_error Optional[bool]: A boolean indicating if an error is retryable.
            Defaults to False.

    Raises:
        google.auth.exceptions.RefreshError: The errors contained in response_data.
    """

    retryable_error = retryable_error if retryable_error else False

    if isinstance(response_data, six.string_types):
        raise exceptions.RefreshError(response_data, retryable=retryable_error)
    try:
        error_details = "{}: {}".format(
            response_data["error"], response_data.get("error_description")
        )
    # If no details could be extracted, use the response data.
    except (KeyError, ValueError):
        error_details = json.dumps(response_data)

    raise exceptions.RefreshError(
        error_details, response_data, retryable=retryable_error
    )


def _can_retry(status_code, response_data):
    """Checks if a request can be retried by inspecting the status code
    and response body of the request.

    Args:
        status_code (int): The response status code.
        response_data (Mapping | str): The decoded response data.

    Returns:
      bool: True if the response is retryable. False otherwise.
    """
    if status_code in transport.DEFAULT_RETRYABLE_STATUS_CODES:
        return True

    try:
        # For a failed response, response_body could be a string
        error_desc = response_data.get("error_description") or ""
        error_code = response_data.get("error") or ""

        if not isinstance(error_code, six.string_types) or not isinstance(
            error_desc, six.string_types
        ):
            return False

        # Per Oauth 2.0 RFC https://www.rfc-editor.org/rfc/rfc6749.html#section-4.1.2.1
        # This is needed because a redirect will not return a 500 status code.
        retryable_error_descriptions = {
            "internal_failure",
            "server_error",
            "temporarily_unavailable",
        }

        if any(e in retryable_error_descriptions for e in (error_code, error_desc)):
            return True

    except AttributeError:
        pass

    return False


def _parse_expiry(response_data):
    """Parses the expiry field from a response into a datetime.

    Args:
        response_data (Mapping): The JSON-parsed response data.

    Returns:
        Optional[datetime]: The expiration or ``None`` if no expiration was
            specified.
    """
    expires_in = response_data.get("expires_in", None)

    if expires_in is not None:
        # Some services do not respect the OAUTH2.0 RFC and send expires_in as a
        # JSON String.
        if isinstance(expires_in, str):
            expires_in = int(expires_in)

        return _helpers.utcnow() + datetime.timedelta(seconds=expires_in)
    else:
        return None


def _token_endpoint_request_no_throw(
    request,
    token_uri,
    body,
    access_token=None,
    use_json=False,
    can_retry=True,
    **kwargs
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
        kwargs: Additional arguments passed on to the request method. The
            kwargs will be passed to `requests.request` method, see:
            https://docs.python-requests.org/en/latest/api/#requests.request.
            For example, you can use `cert=("cert_pem_path", "key_pem_path")`
            to set up client side SSL certificate, and use
            `verify="ca_bundle_path"` to set up the CA certificates for sever
            side SSL certificate verification.

    Returns:
        Tuple(bool, Mapping[str, str], Optional[bool]): A boolean indicating
          if the request is successful, a mapping for the JSON-decoded response
          data and in the case of an error a boolean indicating if the error
          is retryable.
    """
    if use_json:
        headers = {"Content-Type": _JSON_CONTENT_TYPE}
        body = json.dumps(body).encode("utf-8")
    else:
        headers = {"Content-Type": _URLENCODED_CONTENT_TYPE}
        body = urllib.parse.urlencode(body).encode("utf-8")

    if access_token:
        headers["Authorization"] = "Bearer {}".format(access_token)

    def _perform_request():
        response = request(
            method="POST", url=token_uri, headers=headers, body=body, **kwargs
        )
        response_body = (
            response.data.decode("utf-8")
            if hasattr(response.data, "decode")
            else response.data
        )
        response_data = ""
        try:
            # response_body should be a JSON
            response_data = json.loads(response_body)
        except ValueError:
            response_data = response_body

        if response.status == http_client.OK:
            return True, response_data, None

        retryable_error = _can_retry(
            status_code=response.status, response_data=response_data
        )

        return False, response_data, retryable_error

    request_succeeded, response_data, retryable_error = _perform_request()

    if request_succeeded or not retryable_error or not can_retry:
        return request_succeeded, response_data, retryable_error

    retries = _exponential_backoff.ExponentialBackoff()
    for _ in retries:
        request_succeeded, response_data, retryable_error = _perform_request()
        if request_succeeded or not retryable_error:
            return request_succeeded, response_data, retryable_error

    return False, response_data, retryable_error


def _token_endpoint_request(
    request,
    token_uri,
    body,
    access_token=None,
    use_json=False,
    can_retry=True,
    **kwargs
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
        kwargs: Additional arguments passed on to the request method. The
            kwargs will be passed to `requests.request` method, see:
            https://docs.python-requests.org/en/latest/api/#requests.request.
            For example, you can use `cert=("cert_pem_path", "key_pem_path")`
            to set up client side SSL certificate, and use
            `verify="ca_bundle_path"` to set up the CA certificates for sever
            side SSL certificate verification.

    Returns:
        Mapping[str, str]: The JSON-decoded response data.

    Raises:
        google.auth.exceptions.RefreshError: If the token endpoint returned
            an error.
    """

    response_status_ok, response_data, retryable_error = _token_endpoint_request_no_throw(
        request,
        token_uri,
        body,
        access_token=access_token,
        use_json=use_json,
        can_retry=can_retry,
        **kwargs
    )
    if not response_status_ok:
        _handle_error_response(response_data, retryable_error)
    return response_data


def jwt_grant(request, token_uri, assertion, can_retry=True):
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
    body = {"assertion": assertion, "grant_type": _JWT_GRANT_TYPE}

    response_data = _token_endpoint_request(
        request, token_uri, body, can_retry=can_retry
    )

    try:
        access_token = response_data["access_token"]
    except KeyError as caught_exc:
        new_exc = exceptions.RefreshError(
            "No access token in response.", response_data, retryable=False
        )
        six.raise_from(new_exc, caught_exc)

    expiry = _parse_expiry(response_data)

    return access_token, expiry, response_data


def call_iam_generate_id_token_endpoint(request, signer_email, audience, access_token):
    """Call iam.generateIdToken endpoint to get ID token.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        signer_email (str): The signer email used to form the IAM
            generateIdToken endpoint.
        audience (str): The audience for the ID token.
        access_token (str): The access token used to call the IAM endpoint.

    Returns:
        Tuple[str, datetime]: The ID token and expiration.
    """
    body = {"audience": audience, "includeEmail": "true"}

    response_data = _token_endpoint_request(
        request,
        _IAM_IDTOKEN_ENDPOINT.format(signer_email),
        body,
        access_token=access_token,
        use_json=True,
    )

    try:
        id_token = response_data["token"]
    except KeyError as caught_exc:
        new_exc = exceptions.RefreshError(
            "No ID token in response.", response_data, retryable=False
        )
        six.raise_from(new_exc, caught_exc)

    payload = jwt.decode(id_token, verify=False)
    expiry = datetime.datetime.utcfromtimestamp(payload["exp"])

    return id_token, expiry


def id_token_jwt_grant(request, token_uri, assertion, can_retry=True):
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
    body = {"assertion": assertion, "grant_type": _JWT_GRANT_TYPE}

    response_data = _token_endpoint_request(
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


def _handle_refresh_grant_response(response_data, refresh_token):
    """Extract tokens from refresh grant response.

    Args:
        response_data (Mapping[str, str]): Refresh grant response data.
        refresh_token (str): Current refresh token.

    Returns:
        Tuple[str, str, Optional[datetime], Mapping[str, str]]: The access token,
            refresh token, expiration, and additional data returned by the token
            endpoint. If response_data doesn't have refresh token, then the current
            refresh token will be returned.

    Raises:
        google.auth.exceptions.RefreshError: If the token endpoint returned
            an error.
    """
    try:
        access_token = response_data["access_token"]
    except KeyError as caught_exc:
        new_exc = exceptions.RefreshError(
            "No access token in response.", response_data, retryable=False
        )
        six.raise_from(new_exc, caught_exc)

    refresh_token = response_data.get("refresh_token", refresh_token)
    expiry = _parse_expiry(response_data)

    return access_token, refresh_token, expiry, response_data


def refresh_grant(
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
        Tuple[str, str, Optional[datetime], Mapping[str, str]]: The access
            token, new or current refresh token, expiration, and additional data
            returned by the token endpoint.

    Raises:
        google.auth.exceptions.RefreshError: If the token endpoint returned
            an error.

    .. _rfc6748 section 6: https://tools.ietf.org/html/rfc6749#section-6
    """
    body = {
        "grant_type": _REFRESH_GRANT_TYPE,
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }
    if scopes:
        body["scope"] = " ".join(scopes)
    if rapt_token:
        body["rapt"] = rapt_token

    response_data = _token_endpoint_request(
        request, token_uri, body, can_retry=can_retry
    )
    return _handle_refresh_grant_response(response_data, refresh_token)
