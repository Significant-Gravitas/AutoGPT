# Copyright 2018 Google Inc.
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

"""Google Cloud Impersonated credentials.

This module provides authentication for applications where local credentials
impersonates a remote service account using `IAM Credentials API`_.

This class can be used to impersonate a service account as long as the original
Credential object has the "Service Account Token Creator" role on the target
service account.

    .. _IAM Credentials API:
        https://cloud.google.com/iam/credentials/reference/rest/
"""

import base64
import copy
from datetime import datetime
import json

import six
from six.moves import http_client

from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import jwt

_DEFAULT_TOKEN_LIFETIME_SECS = 3600  # 1 hour in seconds

_IAM_SCOPE = ["https://www.googleapis.com/auth/iam"]

_IAM_ENDPOINT = (
    "https://iamcredentials.googleapis.com/v1/projects/-"
    + "/serviceAccounts/{}:generateAccessToken"
)

_IAM_SIGN_ENDPOINT = (
    "https://iamcredentials.googleapis.com/v1/projects/-"
    + "/serviceAccounts/{}:signBlob"
)

_IAM_IDTOKEN_ENDPOINT = (
    "https://iamcredentials.googleapis.com/v1/"
    + "projects/-/serviceAccounts/{}:generateIdToken"
)

_REFRESH_ERROR = "Unable to acquire impersonated credentials"

_DEFAULT_TOKEN_LIFETIME_SECS = 3600  # 1 hour in seconds

_DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"


def _make_iam_token_request(
    request, principal, headers, body, iam_endpoint_override=None
):
    """Makes a request to the Google Cloud IAM service for an access token.
    Args:
        request (Request): The Request object to use.
        principal (str): The principal to request an access token for.
        headers (Mapping[str, str]): Map of headers to transmit.
        body (Mapping[str, str]): JSON Payload body for the iamcredentials
            API call.
        iam_endpoint_override (Optiona[str]): The full IAM endpoint override
            with the target_principal embedded. This is useful when supporting
            impersonation with regional endpoints.

    Raises:
        google.auth.exceptions.TransportError: Raised if there is an underlying
            HTTP connection error
        google.auth.exceptions.RefreshError: Raised if the impersonated
            credentials are not available.  Common reasons are
            `iamcredentials.googleapis.com` is not enabled or the
            `Service Account Token Creator` is not assigned
    """
    iam_endpoint = iam_endpoint_override or _IAM_ENDPOINT.format(principal)

    body = json.dumps(body).encode("utf-8")

    response = request(url=iam_endpoint, method="POST", headers=headers, body=body)

    # support both string and bytes type response.data
    response_body = (
        response.data.decode("utf-8")
        if hasattr(response.data, "decode")
        else response.data
    )

    if response.status != http_client.OK:
        raise exceptions.RefreshError(_REFRESH_ERROR, response_body)

    try:
        token_response = json.loads(response_body)
        token = token_response["accessToken"]
        expiry = datetime.strptime(token_response["expireTime"], "%Y-%m-%dT%H:%M:%SZ")

        return token, expiry

    except (KeyError, ValueError) as caught_exc:
        new_exc = exceptions.RefreshError(
            "{}: No access token or invalid expiration in response.".format(
                _REFRESH_ERROR
            ),
            response_body,
        )
        six.raise_from(new_exc, caught_exc)


class Credentials(
    credentials.Scoped, credentials.CredentialsWithQuotaProject, credentials.Signing
):
    """This module defines impersonated credentials which are essentially
    impersonated identities.

    Impersonated Credentials allows credentials issued to a user or
    service account to impersonate another. The target service account must
    grant the originating credential principal the
    `Service Account Token Creator`_ IAM role:

    For more information about Token Creator IAM role and
    IAMCredentials API, see
    `Creating Short-Lived Service Account Credentials`_.

    .. _Service Account Token Creator:
        https://cloud.google.com/iam/docs/service-accounts#the_service_account_token_creator_role

    .. _Creating Short-Lived Service Account Credentials:
        https://cloud.google.com/iam/docs/creating-short-lived-service-account-credentials

    Usage:

    First grant source_credentials the `Service Account Token Creator`
    role on the target account to impersonate.   In this example, the
    service account represented by svc_account.json has the
    token creator role on
    `impersonated-account@_project_.iam.gserviceaccount.com`.

    Enable the IAMCredentials API on the source project:
    `gcloud services enable iamcredentials.googleapis.com`.

    Initialize a source credential which does not have access to
    list bucket::

        from google.oauth2 import service_account

        target_scopes = [
            'https://www.googleapis.com/auth/devstorage.read_only']

        source_credentials = (
            service_account.Credentials.from_service_account_file(
                '/path/to/svc_account.json',
                scopes=target_scopes))

    Now use the source credentials to acquire credentials to impersonate
    another service account::

        from google.auth import impersonated_credentials

        target_credentials = impersonated_credentials.Credentials(
          source_credentials=source_credentials,
          target_principal='impersonated-account@_project_.iam.gserviceaccount.com',
          target_scopes = target_scopes,
          lifetime=500)

    Resource access is granted::

        client = storage.Client(credentials=target_credentials)
        buckets = client.list_buckets(project='your_project')
        for bucket in buckets:
          print(bucket.name)
    """

    def __init__(
        self,
        source_credentials,
        target_principal,
        target_scopes,
        delegates=None,
        lifetime=_DEFAULT_TOKEN_LIFETIME_SECS,
        quota_project_id=None,
        iam_endpoint_override=None,
    ):
        """
        Args:
            source_credentials (google.auth.Credentials): The source credential
                used as to acquire the impersonated credentials.
            target_principal (str): The service account to impersonate.
            target_scopes (Sequence[str]): Scopes to request during the
                authorization grant.
            delegates (Sequence[str]): The chained list of delegates required
                to grant the final access_token.  If set, the sequence of
                identities must have "Service Account Token Creator" capability
                granted to the prceeding identity.  For example, if set to
                [serviceAccountB, serviceAccountC], the source_credential
                must have the Token Creator role on serviceAccountB.
                serviceAccountB must have the Token Creator on
                serviceAccountC.
                Finally, C must have Token Creator on target_principal.
                If left unset, source_credential must have that role on
                target_principal.
            lifetime (int): Number of seconds the delegated credential should
                be valid for (upto 3600).
            quota_project_id (Optional[str]): The project ID used for quota and billing.
                This project may be different from the project used to
                create the credentials.
            iam_endpoint_override (Optiona[str]): The full IAM endpoint override
                with the target_principal embedded. This is useful when supporting
                impersonation with regional endpoints.
        """

        super(Credentials, self).__init__()

        self._source_credentials = copy.copy(source_credentials)
        # Service account source credentials must have the _IAM_SCOPE
        # added to refresh correctly. User credentials cannot have
        # their original scopes modified.
        if isinstance(self._source_credentials, credentials.Scoped):
            self._source_credentials = self._source_credentials.with_scopes(_IAM_SCOPE)
        self._target_principal = target_principal
        self._target_scopes = target_scopes
        self._delegates = delegates
        self._lifetime = lifetime or _DEFAULT_TOKEN_LIFETIME_SECS
        self.token = None
        self.expiry = _helpers.utcnow()
        self._quota_project_id = quota_project_id
        self._iam_endpoint_override = iam_endpoint_override

    @_helpers.copy_docstring(credentials.Credentials)
    def refresh(self, request):
        self._update_token(request)

    def _update_token(self, request):
        """Updates credentials with a new access_token representing
        the impersonated account.

        Args:
            request (google.auth.transport.requests.Request): Request object
                to use for refreshing credentials.
        """

        # Refresh our source credentials if it is not valid.
        if not self._source_credentials.valid:
            self._source_credentials.refresh(request)

        body = {
            "delegates": self._delegates,
            "scope": self._target_scopes,
            "lifetime": str(self._lifetime) + "s",
        }

        headers = {"Content-Type": "application/json"}

        # Apply the source credentials authentication info.
        self._source_credentials.apply(headers)

        self.token, self.expiry = _make_iam_token_request(
            request=request,
            principal=self._target_principal,
            headers=headers,
            body=body,
            iam_endpoint_override=self._iam_endpoint_override,
        )

    def sign_bytes(self, message):
        from google.auth.transport.requests import AuthorizedSession

        iam_sign_endpoint = _IAM_SIGN_ENDPOINT.format(self._target_principal)

        body = {
            "payload": base64.b64encode(message).decode("utf-8"),
            "delegates": self._delegates,
        }

        headers = {"Content-Type": "application/json"}

        authed_session = AuthorizedSession(self._source_credentials)

        try:
            response = authed_session.post(
                url=iam_sign_endpoint, headers=headers, json=body
            )
        finally:
            authed_session.close()

        if response.status_code != http_client.OK:
            raise exceptions.TransportError(
                "Error calling sign_bytes: {}".format(response.json())
            )

        return base64.b64decode(response.json()["signedBlob"])

    @property
    def signer_email(self):
        return self._target_principal

    @property
    def service_account_email(self):
        return self._target_principal

    @property
    def signer(self):
        return self

    @property
    def requires_scopes(self):
        return not self._target_scopes

    @_helpers.copy_docstring(credentials.CredentialsWithQuotaProject)
    def with_quota_project(self, quota_project_id):
        return self.__class__(
            self._source_credentials,
            target_principal=self._target_principal,
            target_scopes=self._target_scopes,
            delegates=self._delegates,
            lifetime=self._lifetime,
            quota_project_id=quota_project_id,
            iam_endpoint_override=self._iam_endpoint_override,
        )

    @_helpers.copy_docstring(credentials.Scoped)
    def with_scopes(self, scopes, default_scopes=None):
        return self.__class__(
            self._source_credentials,
            target_principal=self._target_principal,
            target_scopes=scopes or default_scopes,
            delegates=self._delegates,
            lifetime=self._lifetime,
            quota_project_id=self._quota_project_id,
            iam_endpoint_override=self._iam_endpoint_override,
        )


class IDTokenCredentials(credentials.CredentialsWithQuotaProject):
    """Open ID Connect ID Token-based service account credentials.

    """

    def __init__(
        self,
        target_credentials,
        target_audience=None,
        include_email=False,
        quota_project_id=None,
    ):
        """
        Args:
            target_credentials (google.auth.Credentials): The target
                credential used as to acquire the id tokens for.
            target_audience (string): Audience to issue the token for.
            include_email (bool): Include email in IdToken
            quota_project_id (Optional[str]):  The project ID used for
                quota and billing.
        """
        super(IDTokenCredentials, self).__init__()

        if not isinstance(target_credentials, Credentials):
            raise exceptions.GoogleAuthError(
                "Provided Credential must be " "impersonated_credentials"
            )
        self._target_credentials = target_credentials
        self._target_audience = target_audience
        self._include_email = include_email
        self._quota_project_id = quota_project_id

    def from_credentials(self, target_credentials, target_audience=None):
        return self.__class__(
            target_credentials=target_credentials,
            target_audience=target_audience,
            include_email=self._include_email,
            quota_project_id=self._quota_project_id,
        )

    def with_target_audience(self, target_audience):
        return self.__class__(
            target_credentials=self._target_credentials,
            target_audience=target_audience,
            include_email=self._include_email,
            quota_project_id=self._quota_project_id,
        )

    def with_include_email(self, include_email):
        return self.__class__(
            target_credentials=self._target_credentials,
            target_audience=self._target_audience,
            include_email=include_email,
            quota_project_id=self._quota_project_id,
        )

    @_helpers.copy_docstring(credentials.CredentialsWithQuotaProject)
    def with_quota_project(self, quota_project_id):
        return self.__class__(
            target_credentials=self._target_credentials,
            target_audience=self._target_audience,
            include_email=self._include_email,
            quota_project_id=quota_project_id,
        )

    @_helpers.copy_docstring(credentials.Credentials)
    def refresh(self, request):
        from google.auth.transport.requests import AuthorizedSession

        iam_sign_endpoint = _IAM_IDTOKEN_ENDPOINT.format(
            self._target_credentials.signer_email
        )

        body = {
            "audience": self._target_audience,
            "delegates": self._target_credentials._delegates,
            "includeEmail": self._include_email,
        }

        headers = {"Content-Type": "application/json"}

        authed_session = AuthorizedSession(
            self._target_credentials._source_credentials, auth_request=request
        )

        response = authed_session.post(
            url=iam_sign_endpoint,
            headers=headers,
            data=json.dumps(body).encode("utf-8"),
        )

        id_token = response.json()["token"]
        self.token = id_token
        self.expiry = datetime.fromtimestamp(jwt.decode(id_token, verify=False)["exp"])
