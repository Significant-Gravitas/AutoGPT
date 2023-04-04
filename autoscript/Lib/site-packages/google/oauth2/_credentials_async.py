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

"""OAuth 2.0 Async Credentials.

This module provides credentials based on OAuth 2.0 access and refresh tokens.
These credentials usually access resources on behalf of a user (resource
owner).

Specifically, this is intended to use access tokens acquired using the
`Authorization Code grant`_ and can refresh those tokens using a
optional `refresh token`_.

Obtaining the initial access and refresh token is outside of the scope of this
module. Consult `rfc6749 section 4.1`_ for complete details on the
Authorization Code grant flow.

.. _Authorization Code grant: https://tools.ietf.org/html/rfc6749#section-1.3.1
.. _refresh token: https://tools.ietf.org/html/rfc6749#section-6
.. _rfc6749 section 4.1: https://tools.ietf.org/html/rfc6749#section-4.1
"""

from google.auth import _credentials_async as credentials
from google.auth import _helpers
from google.auth import exceptions
from google.oauth2 import _reauth_async as reauth
from google.oauth2 import credentials as oauth2_credentials


class Credentials(oauth2_credentials.Credentials):
    """Credentials using OAuth 2.0 access and refresh tokens.

    The credentials are considered immutable. If you want to modify the
    quota project, use :meth:`with_quota_project` or ::

        credentials = credentials.with_quota_project('myproject-123)
    """

    @_helpers.copy_docstring(credentials.Credentials)
    async def refresh(self, request):
        if (
            self._refresh_token is None
            or self._token_uri is None
            or self._client_id is None
            or self._client_secret is None
        ):
            raise exceptions.RefreshError(
                "The credentials do not contain the necessary fields need to "
                "refresh the access token. You must specify refresh_token, "
                "token_uri, client_id, and client_secret."
            )

        (
            access_token,
            refresh_token,
            expiry,
            grant_response,
            rapt_token,
        ) = await reauth.refresh_grant(
            request,
            self._token_uri,
            self._refresh_token,
            self._client_id,
            self._client_secret,
            scopes=self._scopes,
            rapt_token=self._rapt_token,
            enable_reauth_refresh=self._enable_reauth_refresh,
        )

        self.token = access_token
        self.expiry = expiry
        self._refresh_token = refresh_token
        self._id_token = grant_response.get("id_token")
        self._rapt_token = rapt_token

        if self._scopes and "scope" in grant_response:
            requested_scopes = frozenset(self._scopes)
            granted_scopes = frozenset(grant_response["scope"].split())
            scopes_requested_but_not_granted = requested_scopes - granted_scopes
            if scopes_requested_but_not_granted:
                raise exceptions.RefreshError(
                    "Not all requested scopes were granted by the "
                    "authorization server, missing scopes {}.".format(
                        ", ".join(scopes_requested_but_not_granted)
                    )
                )


class UserAccessTokenCredentials(oauth2_credentials.UserAccessTokenCredentials):
    """Access token credentials for user account.

    Obtain the access token for a given user account or the current active
    user account with the ``gcloud auth print-access-token`` command.

    Args:
        account (Optional[str]): Account to get the access token for. If not
            specified, the current active account will be used.
        quota_project_id (Optional[str]): The project ID used for quota
            and billing.

    """
