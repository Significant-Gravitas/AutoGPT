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

"""Service Accounts: JSON Web Token (JWT) Profile for OAuth 2.0

This module implements the JWT Profile for OAuth 2.0 Authorization Grants
as defined by `RFC 7523`_ with particular support for how this RFC is
implemented in Google's infrastructure. Google refers to these credentials
as *Service Accounts*.

Service accounts are used for server-to-server communication, such as
interactions between a web application server and a Google service. The
service account belongs to your application instead of to an individual end
user. In contrast to other OAuth 2.0 profiles, no users are involved and your
application "acts" as the service account.

Typically an application uses a service account when the application uses
Google APIs to work with its own data rather than a user's data. For example,
an application that uses Google Cloud Datastore for data persistence would use
a service account to authenticate its calls to the Google Cloud Datastore API.
However, an application that needs to access a user's Drive documents would
use the normal OAuth 2.0 profile.

Additionally, Google Apps domain administrators can grant service accounts
`domain-wide delegation`_ authority to access user data on behalf of users in
the domain.

This profile uses a JWT to acquire an OAuth 2.0 access token. The JWT is used
in place of the usual authorization token returned during the standard
OAuth 2.0 Authorization Code grant. The JWT is only used for this purpose, as
the acquired access token is used as the bearer token when making requests
using these credentials.

This profile differs from normal OAuth 2.0 profile because no user consent
step is required. The use of the private key allows this profile to assert
identity directly.

This profile also differs from the :mod:`google.auth.jwt` authentication
because the JWT credentials use the JWT directly as the bearer token. This
profile instead only uses the JWT to obtain an OAuth 2.0 access token. The
obtained OAuth 2.0 access token is used as the bearer token.

Domain-wide delegation
----------------------

Domain-wide delegation allows a service account to access user data on
behalf of any user in a Google Apps domain without consent from the user.
For example, an application that uses the Google Calendar API to add events to
the calendars of all users in a Google Apps domain would use a service account
to access the Google Calendar API on behalf of users.

The Google Apps administrator must explicitly authorize the service account to
do this. This authorization step is referred to as "delegating domain-wide
authority" to a service account.

You can use domain-wise delegation by creating a set of credentials with a
specific subject using :meth:`~Credentials.with_subject`.

.. _RFC 7523: https://tools.ietf.org/html/rfc7523
"""

import copy
import datetime

from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import credentials
from google.auth import jwt
from google.oauth2 import _client

_DEFAULT_TOKEN_LIFETIME_SECS = 3600  # 1 hour in seconds
_GOOGLE_OAUTH2_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"


class Credentials(
    credentials.Signing,
    credentials.Scoped,
    credentials.CredentialsWithQuotaProject,
    credentials.CredentialsWithTokenUri,
):
    """Service account credentials

    Usually, you'll create these credentials with one of the helper
    constructors. To create credentials using a Google service account
    private key JSON file::

        credentials = service_account.Credentials.from_service_account_file(
            'service-account.json')

    Or if you already have the service account file loaded::

        service_account_info = json.load(open('service_account.json'))
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info)

    Both helper methods pass on arguments to the constructor, so you can
    specify additional scopes and a subject if necessary::

        credentials = service_account.Credentials.from_service_account_file(
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

    def __init__(
        self,
        signer,
        service_account_email,
        token_uri,
        scopes=None,
        default_scopes=None,
        subject=None,
        project_id=None,
        quota_project_id=None,
        additional_claims=None,
        always_use_jwt_access=False,
    ):
        """
        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            service_account_email (str): The service account's email.
            scopes (Sequence[str]): User-defined scopes to request during the
                authorization grant.
            default_scopes (Sequence[str]): Default scopes passed by a
                Google client library. Use 'scopes' for user-defined scopes.
            token_uri (str): The OAuth 2.0 Token URI.
            subject (str): For domain-wide delegation, the email address of the
                user to for which to request delegated access.
            project_id  (str): Project ID associated with the service account
                credential.
            quota_project_id (Optional[str]): The project ID used for quota and
                billing.
            additional_claims (Mapping[str, str]): Any additional claims for
                the JWT assertion used in the authorization grant.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be always used.

        .. note:: Typically one of the helper constructors
            :meth:`from_service_account_file` or
            :meth:`from_service_account_info` are used instead of calling the
            constructor directly.
        """
        super(Credentials, self).__init__()

        self._scopes = scopes
        self._default_scopes = default_scopes
        self._signer = signer
        self._service_account_email = service_account_email
        self._subject = subject
        self._project_id = project_id
        self._quota_project_id = quota_project_id
        self._token_uri = token_uri
        self._always_use_jwt_access = always_use_jwt_access

        self._jwt_credentials = None

        if additional_claims is not None:
            self._additional_claims = additional_claims
        else:
            self._additional_claims = {}

    @classmethod
    def _from_signer_and_info(cls, signer, info, **kwargs):
        """Creates a Credentials instance from a signer and service account
        info.

        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            info (Mapping[str, str]): The service account info.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.Credentials: The constructed credentials.

        Raises:
            ValueError: If the info is not in the expected format.
        """
        return cls(
            signer,
            service_account_email=info["client_email"],
            token_uri=info["token_uri"],
            project_id=info.get("project_id"),
            **kwargs
        )

    @classmethod
    def from_service_account_info(cls, info, **kwargs):
        """Creates a Credentials instance from parsed service account info.

        Args:
            info (Mapping[str, str]): The service account info in Google
                format.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.service_account.Credentials: The constructed
                credentials.

        Raises:
            ValueError: If the info is not in the expected format.
        """
        signer = _service_account_info.from_dict(
            info, require=["client_email", "token_uri"]
        )
        return cls._from_signer_and_info(signer, info, **kwargs)

    @classmethod
    def from_service_account_file(cls, filename, **kwargs):
        """Creates a Credentials instance from a service account json file.

        Args:
            filename (str): The path to the service account json file.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.service_account.Credentials: The constructed
                credentials.
        """
        info, signer = _service_account_info.from_filename(
            filename, require=["client_email", "token_uri"]
        )
        return cls._from_signer_and_info(signer, info, **kwargs)

    @property
    def service_account_email(self):
        """The service account email."""
        return self._service_account_email

    @property
    def project_id(self):
        """Project ID associated with this credential."""
        return self._project_id

    @property
    def requires_scopes(self):
        """Checks if the credentials requires scopes.

        Returns:
            bool: True if there are no scopes set otherwise False.
        """
        return True if not self._scopes else False

    @_helpers.copy_docstring(credentials.Scoped)
    def with_scopes(self, scopes, default_scopes=None):
        return self.__class__(
            self._signer,
            service_account_email=self._service_account_email,
            scopes=scopes,
            default_scopes=default_scopes,
            token_uri=self._token_uri,
            subject=self._subject,
            project_id=self._project_id,
            quota_project_id=self._quota_project_id,
            additional_claims=self._additional_claims.copy(),
            always_use_jwt_access=self._always_use_jwt_access,
        )

    def with_always_use_jwt_access(self, always_use_jwt_access):
        """Create a copy of these credentials with the specified always_use_jwt_access value.

        Args:
            always_use_jwt_access (bool): Whether always use self signed JWT or not.

        Returns:
            google.auth.service_account.Credentials: A new credentials
                instance.
        """
        return self.__class__(
            self._signer,
            service_account_email=self._service_account_email,
            scopes=self._scopes,
            default_scopes=self._default_scopes,
            token_uri=self._token_uri,
            subject=self._subject,
            project_id=self._project_id,
            quota_project_id=self._quota_project_id,
            additional_claims=self._additional_claims.copy(),
            always_use_jwt_access=always_use_jwt_access,
        )

    def with_subject(self, subject):
        """Create a copy of these credentials with the specified subject.

        Args:
            subject (str): The subject claim.

        Returns:
            google.auth.service_account.Credentials: A new credentials
                instance.
        """
        return self.__class__(
            self._signer,
            service_account_email=self._service_account_email,
            scopes=self._scopes,
            default_scopes=self._default_scopes,
            token_uri=self._token_uri,
            subject=subject,
            project_id=self._project_id,
            quota_project_id=self._quota_project_id,
            additional_claims=self._additional_claims.copy(),
            always_use_jwt_access=self._always_use_jwt_access,
        )

    def with_claims(self, additional_claims):
        """Returns a copy of these credentials with modified claims.

        Args:
            additional_claims (Mapping[str, str]): Any additional claims for
                the JWT payload. This will be merged with the current
                additional claims.

        Returns:
            google.auth.service_account.Credentials: A new credentials
                instance.
        """
        new_additional_claims = copy.deepcopy(self._additional_claims)
        new_additional_claims.update(additional_claims or {})

        return self.__class__(
            self._signer,
            service_account_email=self._service_account_email,
            scopes=self._scopes,
            default_scopes=self._default_scopes,
            token_uri=self._token_uri,
            subject=self._subject,
            project_id=self._project_id,
            quota_project_id=self._quota_project_id,
            additional_claims=new_additional_claims,
            always_use_jwt_access=self._always_use_jwt_access,
        )

    @_helpers.copy_docstring(credentials.CredentialsWithQuotaProject)
    def with_quota_project(self, quota_project_id):

        return self.__class__(
            self._signer,
            service_account_email=self._service_account_email,
            default_scopes=self._default_scopes,
            scopes=self._scopes,
            token_uri=self._token_uri,
            subject=self._subject,
            project_id=self._project_id,
            quota_project_id=quota_project_id,
            additional_claims=self._additional_claims.copy(),
            always_use_jwt_access=self._always_use_jwt_access,
        )

    @_helpers.copy_docstring(credentials.CredentialsWithTokenUri)
    def with_token_uri(self, token_uri):

        return self.__class__(
            self._signer,
            service_account_email=self._service_account_email,
            default_scopes=self._default_scopes,
            scopes=self._scopes,
            token_uri=token_uri,
            subject=self._subject,
            project_id=self._project_id,
            quota_project_id=self._quota_project_id,
            additional_claims=self._additional_claims.copy(),
            always_use_jwt_access=self._always_use_jwt_access,
        )

    def _make_authorization_grant_assertion(self):
        """Create the OAuth 2.0 assertion.

        This assertion is used during the OAuth 2.0 grant to acquire an
        access token.

        Returns:
            bytes: The authorization grant assertion.
        """
        now = _helpers.utcnow()
        lifetime = datetime.timedelta(seconds=_DEFAULT_TOKEN_LIFETIME_SECS)
        expiry = now + lifetime

        payload = {
            "iat": _helpers.datetime_to_secs(now),
            "exp": _helpers.datetime_to_secs(expiry),
            # The issuer must be the service account email.
            "iss": self._service_account_email,
            # The audience must be the auth token endpoint's URI
            "aud": _GOOGLE_OAUTH2_TOKEN_ENDPOINT,
            "scope": _helpers.scopes_to_string(self._scopes or ()),
        }

        payload.update(self._additional_claims)

        # The subject can be a user email for domain-wide delegation.
        if self._subject:
            payload.setdefault("sub", self._subject)

        token = jwt.encode(self._signer, payload)

        return token

    @_helpers.copy_docstring(credentials.Credentials)
    def refresh(self, request):
        # Since domain wide delegation doesn't work with self signed JWT. If
        # subject exists, then we should not use self signed JWT.
        if self._subject is None and self._jwt_credentials is not None:
            self._jwt_credentials.refresh(request)
            self.token = self._jwt_credentials.token
            self.expiry = self._jwt_credentials.expiry
        else:
            assertion = self._make_authorization_grant_assertion()
            access_token, expiry, _ = _client.jwt_grant(
                request, self._token_uri, assertion
            )
            self.token = access_token
            self.expiry = expiry

    def _create_self_signed_jwt(self, audience):
        """Create a self-signed JWT from the credentials if requirements are met.

        Args:
            audience (str): The service URL. ``https://[API_ENDPOINT]/``
        """
        # https://google.aip.dev/auth/4111
        if self._always_use_jwt_access:
            if self._scopes:
                self._jwt_credentials = jwt.Credentials.from_signing_credentials(
                    self, None, additional_claims={"scope": " ".join(self._scopes)}
                )
            elif audience:
                self._jwt_credentials = jwt.Credentials.from_signing_credentials(
                    self, audience
                )
            elif self._default_scopes:
                self._jwt_credentials = jwt.Credentials.from_signing_credentials(
                    self,
                    None,
                    additional_claims={"scope": " ".join(self._default_scopes)},
                )
        elif not self._scopes and audience:
            self._jwt_credentials = jwt.Credentials.from_signing_credentials(
                self, audience
            )

    @_helpers.copy_docstring(credentials.Signing)
    def sign_bytes(self, message):
        return self._signer.sign(message)

    @property  # type: ignore
    @_helpers.copy_docstring(credentials.Signing)
    def signer(self):
        return self._signer

    @property  # type: ignore
    @_helpers.copy_docstring(credentials.Signing)
    def signer_email(self):
        return self._service_account_email


class IDTokenCredentials(
    credentials.Signing,
    credentials.CredentialsWithQuotaProject,
    credentials.CredentialsWithTokenUri,
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
            service_account.IDTokenCredentials.from_service_account_file(
                'service-account.json'))


    Or if you already have the service account file loaded::

        service_account_info = json.load(open('service_account.json'))
        credentials = (
            service_account.IDTokenCredentials.from_service_account_info(
                service_account_info))


    Both helper methods pass on arguments to the constructor, so you can
    specify additional scopes and a subject if necessary::

        credentials = (
            service_account.IDTokenCredentials.from_service_account_file(
                'service-account.json',
                scopes=['email'],
                subject='user@example.com'))


    The credentials are considered immutable. If you want to modify the scopes
    or the subject used for delegation, use :meth:`with_scopes` or
    :meth:`with_subject`::

        scoped_credentials = credentials.with_scopes(['email'])
        delegated_credentials = credentials.with_subject(subject)

    """

    def __init__(
        self,
        signer,
        service_account_email,
        token_uri,
        target_audience,
        additional_claims=None,
        quota_project_id=None,
    ):
        """
        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            service_account_email (str): The service account's email.
            token_uri (str): The OAuth 2.0 Token URI.
            target_audience (str): The intended audience for these credentials,
                used when requesting the ID Token. The ID Token's ``aud`` claim
                will be set to this string.
            additional_claims (Mapping[str, str]): Any additional claims for
                the JWT assertion used in the authorization grant.
            quota_project_id (Optional[str]): The project ID used for quota and billing.
        .. note:: Typically one of the helper constructors
            :meth:`from_service_account_file` or
            :meth:`from_service_account_info` are used instead of calling the
            constructor directly.
        """
        super(IDTokenCredentials, self).__init__()
        self._signer = signer
        self._service_account_email = service_account_email
        self._token_uri = token_uri
        self._target_audience = target_audience
        self._quota_project_id = quota_project_id
        self._use_iam_endpoint = False

        if additional_claims is not None:
            self._additional_claims = additional_claims
        else:
            self._additional_claims = {}

    @classmethod
    def _from_signer_and_info(cls, signer, info, **kwargs):
        """Creates a credentials instance from a signer and service account
        info.

        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            info (Mapping[str, str]): The service account info.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.IDTokenCredentials: The constructed credentials.

        Raises:
            ValueError: If the info is not in the expected format.
        """
        kwargs.setdefault("service_account_email", info["client_email"])
        kwargs.setdefault("token_uri", info["token_uri"])
        return cls(signer, **kwargs)

    @classmethod
    def from_service_account_info(cls, info, **kwargs):
        """Creates a credentials instance from parsed service account info.

        Args:
            info (Mapping[str, str]): The service account info in Google
                format.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.service_account.IDTokenCredentials: The constructed
                credentials.

        Raises:
            ValueError: If the info is not in the expected format.
        """
        signer = _service_account_info.from_dict(
            info, require=["client_email", "token_uri"]
        )
        return cls._from_signer_and_info(signer, info, **kwargs)

    @classmethod
    def from_service_account_file(cls, filename, **kwargs):
        """Creates a credentials instance from a service account json file.

        Args:
            filename (str): The path to the service account json file.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.service_account.IDTokenCredentials: The constructed
                credentials.
        """
        info, signer = _service_account_info.from_filename(
            filename, require=["client_email", "token_uri"]
        )
        return cls._from_signer_and_info(signer, info, **kwargs)

    def with_target_audience(self, target_audience):
        """Create a copy of these credentials with the specified target
        audience.

        Args:
            target_audience (str): The intended audience for these credentials,
            used when requesting the ID Token.

        Returns:
            google.auth.service_account.IDTokenCredentials: A new credentials
                instance.
        """
        return self.__class__(
            self._signer,
            service_account_email=self._service_account_email,
            token_uri=self._token_uri,
            target_audience=target_audience,
            additional_claims=self._additional_claims.copy(),
            quota_project_id=self.quota_project_id,
        )

    def _with_use_iam_endpoint(self, use_iam_endpoint):
        """Create a copy of these credentials with the use_iam_endpoint value.

        Args:
            use_iam_endpoint (bool): If True, IAM generateIdToken endpoint will
                be used instead of the token_uri. Note that
                iam.serviceAccountTokenCreator role is required to use the IAM
                endpoint. The default value is False. This feature is currently
                experimental and subject to change without notice.

        Returns:
            google.auth.service_account.IDTokenCredentials: A new credentials
                instance.
        """
        cred = self.__class__(
            self._signer,
            service_account_email=self._service_account_email,
            token_uri=self._token_uri,
            target_audience=self._target_audience,
            additional_claims=self._additional_claims.copy(),
            quota_project_id=self.quota_project_id,
        )
        cred._use_iam_endpoint = use_iam_endpoint
        return cred

    @_helpers.copy_docstring(credentials.CredentialsWithQuotaProject)
    def with_quota_project(self, quota_project_id):
        return self.__class__(
            self._signer,
            service_account_email=self._service_account_email,
            token_uri=self._token_uri,
            target_audience=self._target_audience,
            additional_claims=self._additional_claims.copy(),
            quota_project_id=quota_project_id,
        )

    @_helpers.copy_docstring(credentials.CredentialsWithTokenUri)
    def with_token_uri(self, token_uri):
        return self.__class__(
            self._signer,
            service_account_email=self._service_account_email,
            token_uri=token_uri,
            target_audience=self._target_audience,
            additional_claims=self._additional_claims.copy(),
            quota_project_id=self._quota_project_id,
        )

    def _make_authorization_grant_assertion(self):
        """Create the OAuth 2.0 assertion.

        This assertion is used during the OAuth 2.0 grant to acquire an
        ID token.

        Returns:
            bytes: The authorization grant assertion.
        """
        now = _helpers.utcnow()
        lifetime = datetime.timedelta(seconds=_DEFAULT_TOKEN_LIFETIME_SECS)
        expiry = now + lifetime

        payload = {
            "iat": _helpers.datetime_to_secs(now),
            "exp": _helpers.datetime_to_secs(expiry),
            # The issuer must be the service account email.
            "iss": self.service_account_email,
            # The audience must be the auth token endpoint's URI
            "aud": _GOOGLE_OAUTH2_TOKEN_ENDPOINT,
            # The target audience specifies which service the ID token is
            # intended for.
            "target_audience": self._target_audience,
        }

        payload.update(self._additional_claims)

        token = jwt.encode(self._signer, payload)

        return token

    def _refresh_with_iam_endpoint(self, request):
        """Use IAM generateIdToken endpoint to obtain an ID token.

        It works as follows:

        1. First we create a self signed jwt with
        https://www.googleapis.com/auth/iam being the scope.

        2. Next we use the self signed jwt as the access token, and make a POST
        request to IAM generateIdToken endpoint. The request body is:
            {
                "audience": self._target_audience,
                "includeEmail": "true"
            }
        TODO: add "set_azp_to_email": "true" once it's ready from server side.
        https://github.com/googleapis/google-auth-library-python/issues/1263

        If the request is succesfully, it will return {"token":"the ID token"},
        and we can extract the ID token and compute its expiry.
        """
        jwt_credentials = jwt.Credentials.from_signing_credentials(
            self,
            None,
            additional_claims={"scope": "https://www.googleapis.com/auth/iam"},
        )
        jwt_credentials.refresh(request)
        self.token, self.expiry = _client.call_iam_generate_id_token_endpoint(
            request,
            self.signer_email,
            self._target_audience,
            jwt_credentials.token.decode(),
        )

    @_helpers.copy_docstring(credentials.Credentials)
    def refresh(self, request):
        if self._use_iam_endpoint:
            self._refresh_with_iam_endpoint(request)
        else:
            assertion = self._make_authorization_grant_assertion()
            access_token, expiry, _ = _client.id_token_jwt_grant(
                request, self._token_uri, assertion
            )
            self.token = access_token
            self.expiry = expiry

    @property
    def service_account_email(self):
        """The service account email."""
        return self._service_account_email

    @_helpers.copy_docstring(credentials.Signing)
    def sign_bytes(self, message):
        return self._signer.sign(message)

    @property  # type: ignore
    @_helpers.copy_docstring(credentials.Signing)
    def signer(self):
        return self._signer

    @property  # type: ignore
    @_helpers.copy_docstring(credentials.Signing)
    def signer_email(self):
        return self._service_account_email
