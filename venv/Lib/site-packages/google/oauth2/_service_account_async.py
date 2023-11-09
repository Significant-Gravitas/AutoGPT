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

"""Service Accounts: JSON Web Token (JWT) Profile for OAuth 2.0

NOTE: This file adds asynchronous refresh methods to both credentials
classes, and therefore async/await syntax is required when calling this
method when using service account credentials with asynchronous functionality.
Otherwise, all other methods are inherited from the regular service account
credentials file google.oauth2.service_account

"""

from google.auth import _credentials_async as credentials_async
from google.auth import _helpers
from google.oauth2 import _client_async
from google.oauth2 import service_account


class Credentials(
    service_account.Credentials, credentials_async.Scoped, credentials_async.Credentials
):
    """Service account credentials

    Usually, you'll create these credentials with one of the helper
    constructors. To create credentials using a Google service account
    private key JSON file::

        credentials = _service_account_async.Credentials.from_service_account_file(
            'service-account.json')

    Or if you already have the service account file loaded::

        service_account_info = json.load(open('service_account.json'))
        credentials = _service_account_async.Credentials.from_service_account_info(
            service_account_info)

    Both helper methods pass on arguments to the constructor, so you can
    specify additional scopes and a subject if necessary::

        credentials = _service_account_async.Credentials.from_service_account_file(
            'service-account.json',
            scopes=['email'],
            subject='user@example.com')

    The credentials are considered immutable. If you want to modify the scopes
    or the subject used for delegation, use :meth:`with_scopes` or
    :meth:`with_subject`::

        scoped_credentials = credentials.with_scopes(['email'])
        delegated_credentials = credentials.with_subject(subject)

    To add a quota project, use :meth:`with_quota_project`::

        credentials = credentials.with_quota_project('myproject-123')
    """

    @_helpers.copy_docstring(credentials_async.Credentials)
    async def refresh(self, request):
        assertion = self._make_authorization_grant_assertion()
        access_token, expiry, _ = await _client_async.jwt_grant(
            request, self._token_uri, assertion
        )
        self.token = access_token
        self.expiry = expiry


class IDTokenCredentials(
    service_account.IDTokenCredentials,
    credentials_async.Signing,
    credentials_async.Credentials,
):
    """Open ID Connect ID Token-based service account credentials.

    These credentials are largely similar to :class:`.Credentials`, but instead
    of using an OAuth 2.0 Access Token as the bearer token, they use an Open
    ID Connect ID Token as the bearer token. These credentials are useful when
    communicating to services that require ID Tokens and can not accept access
    tokens.

    Usually, you'll create these credentials with one of the helper
    constructors. To create credentials using a Google service account
    private key JSON file::

        credentials = (
            _service_account_async.IDTokenCredentials.from_service_account_file(
                'service-account.json'))

    Or if you already have the service account file loaded::

        service_account_info = json.load(open('service_account.json'))
        credentials = (
            _service_account_async.IDTokenCredentials.from_service_account_info(
                service_account_info))

    Both helper methods pass on arguments to the constructor, so you can
    specify additional scopes and a subject if necessary::

        credentials = (
            _service_account_async.IDTokenCredentials.from_service_account_file(
                'service-account.json',
                scopes=['email'],
                subject='user@example.com'))

    The credentials are considered immutable. If you want to modify the scopes
    or the subject used for delegation, use :meth:`with_scopes` or
    :meth:`with_subject`::

        scoped_credentials = credentials.with_scopes(['email'])
        delegated_credentials = credentials.with_subject(subject)

    """

    @_helpers.copy_docstring(credentials_async.Credentials)
    async def refresh(self, request):
        assertion = self._make_authorization_grant_assertion()
        access_token, expiry, _ = await _client_async.id_token_jwt_grant(
            request, self._token_uri, assertion
        )
        self.token = access_token
        self.expiry = expiry
