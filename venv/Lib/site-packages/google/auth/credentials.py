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


"""Interfaces for credentials."""

import abc
import os

import six

from google.auth import _helpers, environment_vars
from google.auth import exceptions


@six.add_metaclass(abc.ABCMeta)
class Credentials(object):
    """Base class for all credentials.

    All credentials have a :attr:`token` that is used for authentication and
    may also optionally set an :attr:`expiry` to indicate when the token will
    no longer be valid.

    Most credentials will be :attr:`invalid` until :meth:`refresh` is called.
    Credentials can do this automatically before the first HTTP request in
    :meth:`before_request`.

    Although the token and expiration will change as the credentials are
    :meth:`refreshed <refresh>` and used, credentials should be considered
    immutable. Various credentials will accept configuration such as private
    keys, scopes, and other options. These options are not changeable after
    construction. Some classes will provide mechanisms to copy the credentials
    with modifications such as :meth:`ScopedCredentials.with_scopes`.
    """

    def __init__(self):
        self.token = None
        """str: The bearer token that can be used in HTTP headers to make
        authenticated requests."""
        self.expiry = None
        """Optional[datetime]: When the token expires and is no longer valid.
        If this is None, the token is assumed to never expire."""
        self._quota_project_id = None
        """Optional[str]: Project to use for quota and billing purposes."""

    @property
    def expired(self):
        """Checks if the credentials are expired.

        Note that credentials can be invalid but not expired because
        Credentials with :attr:`expiry` set to None is considered to never
        expire.
        """
        if not self.expiry:
            return False

        # Remove some threshold from expiry to err on the side of reporting
        # expiration early so that we avoid the 401-refresh-retry loop.
        skewed_expiry = self.expiry - _helpers.REFRESH_THRESHOLD
        return _helpers.utcnow() >= skewed_expiry

    @property
    def valid(self):
        """Checks the validity of the credentials.

        This is True if the credentials have a :attr:`token` and the token
        is not :attr:`expired`.
        """
        return self.token is not None and not self.expired

    @property
    def quota_project_id(self):
        """Project to use for quota and billing purposes."""
        return self._quota_project_id

    @abc.abstractmethod
    def refresh(self, request):
        """Refreshes the access token.

        Args:
            request (google.auth.transport.Request): The object used to make
                HTTP requests.

        Raises:
            google.auth.exceptions.RefreshError: If the credentials could
                not be refreshed.
        """
        # pylint: disable=missing-raises-doc
        # (pylint doesn't recognize that this is abstract)
        raise NotImplementedError("Refresh must be implemented")

    def apply(self, headers, token=None):
        """Apply the token to the authentication header.

        Args:
            headers (Mapping): The HTTP request headers.
            token (Optional[str]): If specified, overrides the current access
                token.
        """
        headers["authorization"] = "Bearer {}".format(
            _helpers.from_bytes(token or self.token)
        )
        if self.quota_project_id:
            headers["x-goog-user-project"] = self.quota_project_id

    def before_request(self, request, method, url, headers):
        """Performs credential-specific before request logic.

        Refreshes the credentials if necessary, then calls :meth:`apply` to
        apply the token to the authentication header.

        Args:
            request (google.auth.transport.Request): The object used to make
                HTTP requests.
            method (str): The request's HTTP method or the RPC method being
                invoked.
            url (str): The request's URI or the RPC service's URI.
            headers (Mapping): The request's headers.
        """
        # pylint: disable=unused-argument
        # (Subclasses may use these arguments to ascertain information about
        # the http request.)
        if not self.valid:
            self.refresh(request)
        self.apply(headers)


class CredentialsWithQuotaProject(Credentials):
    """Abstract base for credentials supporting ``with_quota_project`` factory"""

    def with_quota_project(self, quota_project_id):
        """Returns a copy of these credentials with a modified quota project.

        Args:
            quota_project_id (str): The project to use for quota and
                billing purposes

        Returns:
            google.oauth2.credentials.Credentials: A new credentials instance.
        """
        raise NotImplementedError("This credential does not support quota project.")

    def with_quota_project_from_environment(self):
        quota_from_env = os.environ.get(environment_vars.GOOGLE_CLOUD_QUOTA_PROJECT)
        if quota_from_env:
            return self.with_quota_project(quota_from_env)
        return self


class CredentialsWithTokenUri(Credentials):
    """Abstract base for credentials supporting ``with_token_uri`` factory"""

    def with_token_uri(self, token_uri):
        """Returns a copy of these credentials with a modified token uri.

        Args:
            token_uri (str): The uri to use for fetching/exchanging tokens

        Returns:
            google.oauth2.credentials.Credentials: A new credentials instance.
        """
        raise NotImplementedError("This credential does not use token uri.")


class AnonymousCredentials(Credentials):
    """Credentials that do not provide any authentication information.

    These are useful in the case of services that support anonymous access or
    local service emulators that do not use credentials.
    """

    @property
    def expired(self):
        """Returns `False`, anonymous credentials never expire."""
        return False

    @property
    def valid(self):
        """Returns `True`, anonymous credentials are always valid."""
        return True

    def refresh(self, request):
        """Raises :class:``InvalidOperation``, anonymous credentials cannot be
        refreshed."""
        raise exceptions.InvalidOperation("Anonymous credentials cannot be refreshed.")

    def apply(self, headers, token=None):
        """Anonymous credentials do nothing to the request.

        The optional ``token`` argument is not supported.

        Raises:
            google.auth.exceptions.InvalidValue: If a token was specified.
        """
        if token is not None:
            raise exceptions.InvalidValue("Anonymous credentials don't support tokens.")

    def before_request(self, request, method, url, headers):
        """Anonymous credentials do nothing to the request."""


@six.add_metaclass(abc.ABCMeta)
class ReadOnlyScoped(object):
    """Interface for credentials whose scopes can be queried.

    OAuth 2.0-based credentials allow limiting access using scopes as described
    in `RFC6749 Section 3.3`_.
    If a credential class implements this interface then the credentials either
    use scopes in their implementation.

    Some credentials require scopes in order to obtain a token. You can check
    if scoping is necessary with :attr:`requires_scopes`::

        if credentials.requires_scopes:
            # Scoping is required.
            credentials = credentials.with_scopes(scopes=['one', 'two'])

    Credentials that require scopes must either be constructed with scopes::

        credentials = SomeScopedCredentials(scopes=['one', 'two'])

    Or must copy an existing instance using :meth:`with_scopes`::

        scoped_credentials = credentials.with_scopes(scopes=['one', 'two'])

    Some credentials have scopes but do not allow or require scopes to be set,
    these credentials can be used as-is.

    .. _RFC6749 Section 3.3: https://tools.ietf.org/html/rfc6749#section-3.3
    """

    def __init__(self):
        super(ReadOnlyScoped, self).__init__()
        self._scopes = None
        self._default_scopes = None

    @property
    def scopes(self):
        """Sequence[str]: the credentials' current set of scopes."""
        return self._scopes

    @property
    def default_scopes(self):
        """Sequence[str]: the credentials' current set of default scopes."""
        return self._default_scopes

    @abc.abstractproperty
    def requires_scopes(self):
        """True if these credentials require scopes to obtain an access token.
        """
        return False

    def has_scopes(self, scopes):
        """Checks if the credentials have the given scopes.

        .. warning: This method is not guaranteed to be accurate if the
            credentials are :attr:`~Credentials.invalid`.

        Args:
            scopes (Sequence[str]): The list of scopes to check.

        Returns:
            bool: True if the credentials have the given scopes.
        """
        credential_scopes = (
            self._scopes if self._scopes is not None else self._default_scopes
        )
        return set(scopes).issubset(set(credential_scopes or []))


class Scoped(ReadOnlyScoped):
    """Interface for credentials whose scopes can be replaced while copying.

    OAuth 2.0-based credentials allow limiting access using scopes as described
    in `RFC6749 Section 3.3`_.
    If a credential class implements this interface then the credentials either
    use scopes in their implementation.

    Some credentials require scopes in order to obtain a token. You can check
    if scoping is necessary with :attr:`requires_scopes`::

        if credentials.requires_scopes:
            # Scoping is required.
            credentials = credentials.create_scoped(['one', 'two'])

    Credentials that require scopes must either be constructed with scopes::

        credentials = SomeScopedCredentials(scopes=['one', 'two'])

    Or must copy an existing instance using :meth:`with_scopes`::

        scoped_credentials = credentials.with_scopes(scopes=['one', 'two'])

    Some credentials have scopes but do not allow or require scopes to be set,
    these credentials can be used as-is.

    .. _RFC6749 Section 3.3: https://tools.ietf.org/html/rfc6749#section-3.3
    """

    @abc.abstractmethod
    def with_scopes(self, scopes, default_scopes=None):
        """Create a copy of these credentials with the specified scopes.

        Args:
            scopes (Sequence[str]): The list of scopes to attach to the
                current credentials.

        Raises:
            NotImplementedError: If the credentials' scopes can not be changed.
                This can be avoided by checking :attr:`requires_scopes` before
                calling this method.
        """
        raise NotImplementedError("This class does not require scoping.")


def with_scopes_if_required(credentials, scopes, default_scopes=None):
    """Creates a copy of the credentials with scopes if scoping is required.

    This helper function is useful when you do not know (or care to know) the
    specific type of credentials you are using (such as when you use
    :func:`google.auth.default`). This function will call
    :meth:`Scoped.with_scopes` if the credentials are scoped credentials and if
    the credentials require scoping. Otherwise, it will return the credentials
    as-is.

    Args:
        credentials (google.auth.credentials.Credentials): The credentials to
            scope if necessary.
        scopes (Sequence[str]): The list of scopes to use.
        default_scopes (Sequence[str]): Default scopes passed by a
            Google client library. Use 'scopes' for user-defined scopes.

    Returns:
        google.auth.credentials.Credentials: Either a new set of scoped
            credentials, or the passed in credentials instance if no scoping
            was required.
    """
    if isinstance(credentials, Scoped) and credentials.requires_scopes:
        return credentials.with_scopes(scopes, default_scopes=default_scopes)
    else:
        return credentials


@six.add_metaclass(abc.ABCMeta)
class Signing(object):
    """Interface for credentials that can cryptographically sign messages."""

    @abc.abstractmethod
    def sign_bytes(self, message):
        """Signs the given message.

        Args:
            message (bytes): The message to sign.

        Returns:
            bytes: The message's cryptographic signature.
        """
        # pylint: disable=missing-raises-doc,redundant-returns-doc
        # (pylint doesn't recognize that this is abstract)
        raise NotImplementedError("Sign bytes must be implemented.")

    @abc.abstractproperty
    def signer_email(self):
        """Optional[str]: An email address that identifies the signer."""
        # pylint: disable=missing-raises-doc
        # (pylint doesn't recognize that this is abstract)
        raise NotImplementedError("Signer email must be implemented.")

    @abc.abstractproperty
    def signer(self):
        """google.auth.crypt.Signer: The signer used to sign bytes."""
        # pylint: disable=missing-raises-doc
        # (pylint doesn't recognize that this is abstract)
        raise NotImplementedError("Signer must be implemented.")
