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

"""External Account Authorized User Credentials.
This module provides credentials based on OAuth 2.0 access and refresh tokens.
These credentials usually access resources on behalf of a user (resource
owner).

Specifically, these are sourced using external identities via Workforce Identity Federation.

Obtaining the initial access and refresh token can be done through the Google Cloud CLI.

Example credential:
{
  "type": "external_account_authorized_user",
  "audience": "//iam.googleapis.com/locations/global/workforcePools/$WORKFORCE_POOL_ID/providers/$PROVIDER_ID",
  "refresh_token": "refreshToken",
  "token_url": "https://sts.googleapis.com/v1/oauth/token",
  "token_info_url": "https://sts.googleapis.com/v1/instrospect",
  "client_id": "clientId",
  "client_secret": "clientSecret"
}
"""

import datetime
import io
import json

from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
from google.oauth2 import utils

_EXTERNAL_ACCOUNT_AUTHORIZED_USER_JSON_TYPE = "external_account_authorized_user"


class Credentials(
    credentials.CredentialsWithQuotaProject,
    credentials.ReadOnlyScoped,
    credentials.CredentialsWithTokenUri,
):
    """Credentials for External Account Authorized Users.

    This is used to instantiate Credentials for exchanging refresh tokens from
    authorized users for Google access token and authorizing requests to Google
    APIs.

    The credentials are considered immutable. If you want to modify the
    quota project, use `with_quota_project` and if you want to modify the token
    uri, use `with_token_uri`.
    """

    def __init__(
        self,
        token=None,
        expiry=None,
        refresh_token=None,
        audience=None,
        client_id=None,
        client_secret=None,
        token_url=None,
        token_info_url=None,
        revoke_url=None,
        scopes=None,
        quota_project_id=None,
    ):
        """Instantiates a external account authorized user credentials object.

        Args:
        token (str): The OAuth 2.0 access token. Can be None if refresh information
            is provided.
        expiry (datetime.datetime): The optional expiration datetime of the OAuth 2.0 access
            token.
        refresh_token (str): The optional OAuth 2.0 refresh token. If specified,
            credentials can be refreshed.
        audience (str): The optional STS audience which contains the resource name for the workforce
            pool and the provider identifier in that pool.
        client_id (str): The OAuth 2.0 client ID. Must be specified for refresh, can be left as
            None if the token can not be refreshed.
        client_secret (str): The OAuth 2.0 client secret. Must be specified for refresh, can be
            left as None if the token can not be refreshed.
        token_url (str): The optional STS token exchange endpoint for refresh. Must be specified for
            refresh, can be left as None if the token can not be refreshed.
        token_info_url (str): The optional STS endpoint URL for token introspection.
        revoke_url (str): The optional STS endpoint URL for revoking tokens.
        quota_project_id (str): The optional project ID used for quota and billing.
            This project may be different from the project used to
            create the credentials.

        Returns:
            google.auth.external_account_authorized_user.Credentials: The
                constructed credentials.
        """
        super(Credentials, self).__init__()

        self.token = token
        self.expiry = expiry
        self._audience = audience
        self._refresh_token = refresh_token
        self._token_url = token_url
        self._token_info_url = token_info_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._revoke_url = revoke_url
        self._quota_project_id = quota_project_id
        self._scopes = scopes

        if not self.valid and not self.can_refresh:
            raise exceptions.InvalidOperation(
                "Token should be created with fields to make it valid (`token` and "
                "`expiry`), or fields to allow it to refresh (`refresh_token`, "
                "`token_url`, `client_id`, `client_secret`)."
            )

        self._client_auth = None
        if self._client_id:
            self._client_auth = utils.ClientAuthentication(
                utils.ClientAuthType.basic, self._client_id, self._client_secret
            )
        self._sts_client = sts.Client(self._token_url, self._client_auth)

    @property
    def info(self):
        """Generates the serializable dictionary representation of the current
        credentials.

        Returns:
            Mapping: The dictionary representation of the credentials. This is the
                reverse of the "from_info" method defined in this class. It is
                useful for serializing the current credentials so it can deserialized
                later.
        """
        config_info = self.constructor_args()
        config_info.update(type=_EXTERNAL_ACCOUNT_AUTHORIZED_USER_JSON_TYPE)
        if config_info["expiry"]:
            config_info["expiry"] = config_info["expiry"].isoformat() + "Z"

        return {key: value for key, value in config_info.items() if value is not None}

    def constructor_args(self):
        return {
            "audience": self._audience,
            "refresh_token": self._refresh_token,
            "token_url": self._token_url,
            "token_info_url": self._token_info_url,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "token": self.token,
            "expiry": self.expiry,
            "revoke_url": self._revoke_url,
            "scopes": self._scopes,
            "quota_project_id": self._quota_project_id,
        }

    @property
    def scopes(self):
        """Optional[str]: The OAuth 2.0 permission scopes."""
        return self._scopes

    @property
    def requires_scopes(self):
        """ False: OAuth 2.0 credentials have their scopes set when
        the initial token is requested and can not be changed."""
        return False

    @property
    def client_id(self):
        """Optional[str]: The OAuth 2.0 client ID."""
        return self._client_id

    @property
    def client_secret(self):
        """Optional[str]: The OAuth 2.0 client secret."""
        return self._client_secret

    @property
    def audience(self):
        """Optional[str]: The STS audience which contains the resource name for the
            workforce pool and the provider identifier in that pool."""
        return self._audience

    @property
    def refresh_token(self):
        """Optional[str]: The OAuth 2.0 refresh token."""
        return self._refresh_token

    @property
    def token_url(self):
        """Optional[str]: The STS token exchange endpoint for refresh."""
        return self._token_url

    @property
    def token_info_url(self):
        """Optional[str]: The STS endpoint for token info."""
        return self._token_info_url

    @property
    def revoke_url(self):
        """Optional[str]: The STS endpoint for token revocation."""
        return self._revoke_url

    @property
    def is_user(self):
        """ True: This credential always represents a user."""
        return True

    @property
    def can_refresh(self):
        return all(
            (self._refresh_token, self._token_url, self._client_id, self._client_secret)
        )

    def get_project_id(self, request=None):
        """Retrieves the project ID corresponding to the workload identity or workforce pool.
        For workforce pool credentials, it returns the project ID corresponding to
        the workforce_pool_user_project.

        When not determinable, None is returned.

        Args:
            request (google.auth.transport.requests.Request): Request object.
                Unused here, but passed from _default.default().

        Return:
          str: project ID is not determinable for this credential type so it returns None
        """

        return None

    def to_json(self, strip=None):
        """Utility function that creates a JSON representation of this
        credential.
        Args:
            strip (Sequence[str]): Optional list of members to exclude from the
                                   generated JSON.
        Returns:
            str: A JSON representation of this instance. When converted into
            a dictionary, it can be passed to from_info()
            to create a new instance.
        """
        strip = strip if strip else []
        return json.dumps({k: v for (k, v) in self.info.items() if k not in strip})

    def refresh(self, request):
        """Refreshes the access token.

        Args:
            request (google.auth.transport.Request): The object used to make
                HTTP requests.

        Raises:
            google.auth.exceptions.RefreshError: If the credentials could
                not be refreshed.
        """
        if not self.can_refresh:
            raise exceptions.RefreshError(
                "The credentials do not contain the necessary fields need to "
                "refresh the access token. You must specify refresh_token, "
                "token_url, client_id, and client_secret."
            )

        now = _helpers.utcnow()
        response_data = self._make_sts_request(request)

        self.token = response_data.get("access_token")

        lifetime = datetime.timedelta(seconds=response_data.get("expires_in"))
        self.expiry = now + lifetime

        if "refresh_token" in response_data:
            self._refresh_token = response_data["refresh_token"]

    def _make_sts_request(self, request):
        return self._sts_client.refresh_token(request, self._refresh_token)

    @_helpers.copy_docstring(credentials.CredentialsWithQuotaProject)
    def with_quota_project(self, quota_project_id):
        kwargs = self.constructor_args()
        kwargs.update(quota_project_id=quota_project_id)
        return self.__class__(**kwargs)

    @_helpers.copy_docstring(credentials.CredentialsWithTokenUri)
    def with_token_uri(self, token_uri):
        kwargs = self.constructor_args()
        kwargs.update(token_url=token_uri)
        return self.__class__(**kwargs)

    @classmethod
    def from_info(cls, info, **kwargs):
        """Creates a Credentials instance from parsed external account info.

        Args:
            info (Mapping[str, str]): The external account info in Google
                format.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.external_account_authorized_user.Credentials: The
                constructed credentials.

        Raises:
            ValueError: For invalid parameters.
        """
        expiry = info.get("expiry")
        if expiry:
            expiry = datetime.datetime.strptime(
                expiry.rstrip("Z").split(".")[0], "%Y-%m-%dT%H:%M:%S"
            )
        return cls(
            audience=info.get("audience"),
            refresh_token=info.get("refresh_token"),
            token_url=info.get("token_url"),
            token_info_url=info.get("token_info_url"),
            client_id=info.get("client_id"),
            client_secret=info.get("client_secret"),
            token=info.get("token"),
            expiry=expiry,
            revoke_url=info.get("revoke_url"),
            quota_project_id=info.get("quota_project_id"),
            scopes=info.get("scopes"),
            **kwargs
        )

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Creates a Credentials instance from an external account json file.

        Args:
            filename (str): The path to the external account json file.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.external_account_authorized_user.Credentials: The
                constructed credentials.
        """
        with io.open(filename, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            return cls.from_info(data, **kwargs)
