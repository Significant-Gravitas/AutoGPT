# Copyright 2021 Google LLC
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

"""Downscoping with Credential Access Boundaries

This module provides the ability to downscope credentials using
`Downscoping with Credential Access Boundaries`_. This is useful to restrict the
Identity and Access Management (IAM) permissions that a short-lived credential
can use.

To downscope permissions of a source credential, a Credential Access Boundary
that specifies which resources the new credential can access, as well as
an upper bound on the permissions that are available on each resource, has to
be defined. A downscoped credential can then be instantiated using the source
credential and the Credential Access Boundary.

The common pattern of usage is to have a token broker with elevated access
generate these downscoped credentials from higher access source credentials and
pass the downscoped short-lived access tokens to a token consumer via some
secure authenticated channel for limited access to Google Cloud Storage
resources.

For example, a token broker can be set up on a server in a private network.
Various workloads (token consumers) in the same network will send authenticated
requests to that broker for downscoped tokens to access or modify specific google
cloud storage buckets.

The broker will instantiate downscoped credentials instances that can be used to
generate short lived downscoped access tokens that can be passed to the token
consumer. These downscoped access tokens can be injected by the consumer into
google.oauth2.Credentials and used to initialize a storage client instance to
access Google Cloud Storage resources with restricted access.

Note: Only Cloud Storage supports Credential Access Boundaries. Other Google
Cloud services do not support this feature.

.. _Downscoping with Credential Access Boundaries: https://cloud.google.com/iam/docs/downscoping-short-lived-credentials
"""

import datetime

import six

from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts

# The maximum number of access boundary rules a Credential Access Boundary can
# contain.
_MAX_ACCESS_BOUNDARY_RULES_COUNT = 10
# The token exchange grant_type used for exchanging credentials.
_STS_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
# The token exchange requested_token_type. This is always an access_token.
_STS_REQUESTED_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
# The STS token URL used to exchanged a short lived access token for a downscoped one.
_STS_TOKEN_URL = "https://sts.googleapis.com/v1/token"
# The subject token type to use when exchanging a short lived access token for a
# downscoped token.
_STS_SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"


class CredentialAccessBoundary(object):
    """Defines a Credential Access Boundary which contains a list of access boundary
    rules. Each rule contains information on the resource that the rule applies to,
    the upper bound of the permissions that are available on that resource and an
    optional condition to further restrict permissions.
    """

    def __init__(self, rules=[]):
        """Instantiates a Credential Access Boundary. A Credential Access Boundary
        can contain up to 10 access boundary rules.

        Args:
            rules (Sequence[google.auth.downscoped.AccessBoundaryRule]): The list of
                access boundary rules limiting the access that a downscoped credential
                will have.
        Raises:
            InvalidType: If any of the rules are not a valid type.
            InvalidValue: If the provided rules exceed the maximum allowed.
        """
        self.rules = rules

    @property
    def rules(self):
        """Returns the list of access boundary rules defined on the Credential
        Access Boundary.

        Returns:
            Tuple[google.auth.downscoped.AccessBoundaryRule, ...]: The list of access
                boundary rules defined on the Credential Access Boundary. These are returned
                as an immutable tuple to prevent modification.
        """
        return tuple(self._rules)

    @rules.setter
    def rules(self, value):
        """Updates the current rules on the Credential Access Boundary. This will overwrite
        the existing set of rules.

        Args:
            value (Sequence[google.auth.downscoped.AccessBoundaryRule]): The list of
                access boundary rules limiting the access that a downscoped credential
                will have.
        Raises:
            InvalidType: If any of the rules are not a valid type.
            InvalidValue: If the provided rules exceed the maximum allowed.
        """
        if len(value) > _MAX_ACCESS_BOUNDARY_RULES_COUNT:
            raise exceptions.InvalidValue(
                "Credential access boundary rules can have a maximum of {} rules.".format(
                    _MAX_ACCESS_BOUNDARY_RULES_COUNT
                )
            )
        for access_boundary_rule in value:
            if not isinstance(access_boundary_rule, AccessBoundaryRule):
                raise exceptions.InvalidType(
                    "List of rules provided do not contain a valid 'google.auth.downscoped.AccessBoundaryRule'."
                )
        # Make a copy of the original list.
        self._rules = list(value)

    def add_rule(self, rule):
        """Adds a single access boundary rule to the existing rules.

        Args:
            rule (google.auth.downscoped.AccessBoundaryRule): The access boundary rule,
                limiting the access that a downscoped credential will have, to be added to
                the existing rules.
        Raises:
            InvalidType: If any of the rules are not a valid type.
            InvalidValue: If the provided rules exceed the maximum allowed.
        """
        if len(self.rules) == _MAX_ACCESS_BOUNDARY_RULES_COUNT:
            raise exceptions.InvalidValue(
                "Credential access boundary rules can have a maximum of {} rules.".format(
                    _MAX_ACCESS_BOUNDARY_RULES_COUNT
                )
            )
        if not isinstance(rule, AccessBoundaryRule):
            raise exceptions.InvalidType(
                "The provided rule does not contain a valid 'google.auth.downscoped.AccessBoundaryRule'."
            )
        self._rules.append(rule)

    def to_json(self):
        """Generates the dictionary representation of the Credential Access Boundary.
        This uses the format expected by the Security Token Service API as documented in
        `Defining a Credential Access Boundary`_.

        .. _Defining a Credential Access Boundary:
            https://cloud.google.com/iam/docs/downscoping-short-lived-credentials#define-boundary

        Returns:
            Mapping: Credential Access Boundary Rule represented in a dictionary object.
        """
        rules = []
        for access_boundary_rule in self.rules:
            rules.append(access_boundary_rule.to_json())

        return {"accessBoundary": {"accessBoundaryRules": rules}}


class AccessBoundaryRule(object):
    """Defines an access boundary rule which contains information on the resource that
    the rule applies to, the upper bound of the permissions that are available on that
    resource and an optional condition to further restrict permissions.
    """

    def __init__(
        self, available_resource, available_permissions, availability_condition=None
    ):
        """Instantiates a single access boundary rule.

        Args:
            available_resource (str): The full resource name of the Cloud Storage bucket
                that the rule applies to. Use the format
                "//storage.googleapis.com/projects/_/buckets/bucket-name".
            available_permissions (Sequence[str]): A list defining the upper bound that
                the downscoped token will have on the available permissions for the
                resource. Each value is the identifier for an IAM predefined role or
                custom role, with the prefix "inRole:". For example:
                "inRole:roles/storage.objectViewer".
                Only the permissions in these roles will be available.
            availability_condition (Optional[google.auth.downscoped.AvailabilityCondition]):
                Optional condition that restricts the availability of permissions to
                specific Cloud Storage objects.

        Raises:
            InvalidType: If any of the parameters are not of the expected types.
            InvalidValue: If any of the parameters are not of the expected values.
        """
        self.available_resource = available_resource
        self.available_permissions = available_permissions
        self.availability_condition = availability_condition

    @property
    def available_resource(self):
        """Returns the current available resource.

        Returns:
           str: The current available resource.
        """
        return self._available_resource

    @available_resource.setter
    def available_resource(self, value):
        """Updates the current available resource.

        Args:
            value (str): The updated value of the available resource.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not a string.
        """
        if not isinstance(value, six.string_types):
            raise exceptions.InvalidType(
                "The provided available_resource is not a string."
            )
        self._available_resource = value

    @property
    def available_permissions(self):
        """Returns the current available permissions.

        Returns:
           Tuple[str, ...]: The current available permissions. These are returned
               as an immutable tuple to prevent modification.
        """
        return tuple(self._available_permissions)

    @available_permissions.setter
    def available_permissions(self, value):
        """Updates the current available permissions.

        Args:
            value (Sequence[str]): The updated value of the available permissions.

        Raises:
            InvalidType: If the value is not a list of strings.
            InvalidValue: If the value is not valid.
        """
        for available_permission in value:
            if not isinstance(available_permission, six.string_types):
                raise exceptions.InvalidType(
                    "Provided available_permissions are not a list of strings."
                )
            if available_permission.find("inRole:") != 0:
                raise exceptions.InvalidValue(
                    "available_permissions must be prefixed with 'inRole:'."
                )
        # Make a copy of the original list.
        self._available_permissions = list(value)

    @property
    def availability_condition(self):
        """Returns the current availability condition.

        Returns:
           Optional[google.auth.downscoped.AvailabilityCondition]: The current
               availability condition.
        """
        return self._availability_condition

    @availability_condition.setter
    def availability_condition(self, value):
        """Updates the current availability condition.

        Args:
            value (Optional[google.auth.downscoped.AvailabilityCondition]): The updated
                value of the availability condition.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not of type google.auth.downscoped.AvailabilityCondition
                or None.
        """
        if not isinstance(value, AvailabilityCondition) and value is not None:
            raise exceptions.InvalidType(
                "The provided availability_condition is not a 'google.auth.downscoped.AvailabilityCondition' or None."
            )
        self._availability_condition = value

    def to_json(self):
        """Generates the dictionary representation of the access boundary rule.
        This uses the format expected by the Security Token Service API as documented in
        `Defining a Credential Access Boundary`_.

        .. _Defining a Credential Access Boundary:
            https://cloud.google.com/iam/docs/downscoping-short-lived-credentials#define-boundary

        Returns:
            Mapping: The access boundary rule represented in a dictionary object.
        """
        json = {
            "availablePermissions": list(self.available_permissions),
            "availableResource": self.available_resource,
        }
        if self.availability_condition:
            json["availabilityCondition"] = self.availability_condition.to_json()
        return json


class AvailabilityCondition(object):
    """An optional condition that can be used as part of a Credential Access Boundary
    to further restrict permissions."""

    def __init__(self, expression, title=None, description=None):
        """Instantiates an availability condition using the provided expression and
        optional title or description.

        Args:
            expression (str): A condition expression that specifies the Cloud Storage
                objects where permissions are available. For example, this expression
                makes permissions available for objects whose name starts with "customer-a":
                "resource.name.startsWith('projects/_/buckets/example-bucket/objects/customer-a')"
            title (Optional[str]): An optional short string that identifies the purpose of
                the condition.
            description (Optional[str]): Optional details about the purpose of the condition.

        Raises:
            InvalidType: If any of the parameters are not of the expected types.
            InvalidValue: If any of the parameters are not of the expected values.
        """
        self.expression = expression
        self.title = title
        self.description = description

    @property
    def expression(self):
        """Returns the current condition expression.

        Returns:
           str: The current conditon expression.
        """
        return self._expression

    @expression.setter
    def expression(self, value):
        """Updates the current condition expression.

        Args:
            value (str): The updated value of the condition expression.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not of type string.
        """
        if not isinstance(value, six.string_types):
            raise exceptions.InvalidType("The provided expression is not a string.")
        self._expression = value

    @property
    def title(self):
        """Returns the current title.

        Returns:
           Optional[str]: The current title.
        """
        return self._title

    @title.setter
    def title(self, value):
        """Updates the current title.

        Args:
            value (Optional[str]): The updated value of the title.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not of type string or None.
        """
        if not isinstance(value, six.string_types) and value is not None:
            raise exceptions.InvalidType("The provided title is not a string or None.")
        self._title = value

    @property
    def description(self):
        """Returns the current description.

        Returns:
           Optional[str]: The current description.
        """
        return self._description

    @description.setter
    def description(self, value):
        """Updates the current description.

        Args:
            value (Optional[str]): The updated value of the description.

        Raises:
            google.auth.exceptions.InvalidType: If the value is not of type string or None.
        """
        if not isinstance(value, six.string_types) and value is not None:
            raise exceptions.InvalidType(
                "The provided description is not a string or None."
            )
        self._description = value

    def to_json(self):
        """Generates the dictionary representation of the availability condition.
        This uses the format expected by the Security Token Service API as documented in
        `Defining a Credential Access Boundary`_.

        .. _Defining a Credential Access Boundary:
            https://cloud.google.com/iam/docs/downscoping-short-lived-credentials#define-boundary

        Returns:
            Mapping[str, str]: The availability condition represented in a dictionary
                object.
        """
        json = {"expression": self.expression}
        if self.title:
            json["title"] = self.title
        if self.description:
            json["description"] = self.description
        return json


class Credentials(credentials.CredentialsWithQuotaProject):
    """Defines a set of Google credentials that are downscoped from an existing set
    of Google OAuth2 credentials. This is useful to restrict the Identity and Access
    Management (IAM) permissions that a short-lived credential can use.
    The common pattern of usage is to have a token broker with elevated access
    generate these downscoped credentials from higher access source credentials and
    pass the downscoped short-lived access tokens to a token consumer via some
    secure authenticated channel for limited access to Google Cloud Storage
    resources.
    """

    def __init__(
        self, source_credentials, credential_access_boundary, quota_project_id=None
    ):
        """Instantiates a downscoped credentials object using the provided source
        credentials and credential access boundary rules.
        To downscope permissions of a source credential, a Credential Access Boundary
        that specifies which resources the new credential can access, as well as an
        upper bound on the permissions that are available on each resource, has to be
        defined. A downscoped credential can then be instantiated using the source
        credential and the Credential Access Boundary.

        Args:
            source_credentials (google.auth.credentials.Credentials): The source credentials
                to be downscoped based on the provided Credential Access Boundary rules.
            credential_access_boundary (google.auth.downscoped.CredentialAccessBoundary):
                The Credential Access Boundary which contains a list of access boundary
                rules. Each rule contains information on the resource that the rule applies to,
                the upper bound of the permissions that are available on that resource and an
                optional condition to further restrict permissions.
            quota_project_id (Optional[str]): The optional quota project ID.
        Raises:
            google.auth.exceptions.RefreshError: If the source credentials
                return an error on token refresh.
            google.auth.exceptions.OAuthError: If the STS token exchange
                endpoint returned an error during downscoped token generation.
        """

        super(Credentials, self).__init__()
        self._source_credentials = source_credentials
        self._credential_access_boundary = credential_access_boundary
        self._quota_project_id = quota_project_id
        self._sts_client = sts.Client(_STS_TOKEN_URL)

    @_helpers.copy_docstring(credentials.Credentials)
    def refresh(self, request):
        # Generate an access token from the source credentials.
        self._source_credentials.refresh(request)
        now = _helpers.utcnow()
        # Exchange the access token for a downscoped access token.
        response_data = self._sts_client.exchange_token(
            request=request,
            grant_type=_STS_GRANT_TYPE,
            subject_token=self._source_credentials.token,
            subject_token_type=_STS_SUBJECT_TOKEN_TYPE,
            requested_token_type=_STS_REQUESTED_TOKEN_TYPE,
            additional_options=self._credential_access_boundary.to_json(),
        )
        self.token = response_data.get("access_token")
        # For downscoping CAB flow, the STS endpoint may not return the expiration
        # field for some flows. The generated downscoped token should always have
        # the same expiration time as the source credentials. When no expires_in
        # field is returned in the response, we can just get the expiration time
        # from the source credentials.
        if response_data.get("expires_in"):
            lifetime = datetime.timedelta(seconds=response_data.get("expires_in"))
            self.expiry = now + lifetime
        else:
            self.expiry = self._source_credentials.expiry

    @_helpers.copy_docstring(credentials.CredentialsWithQuotaProject)
    def with_quota_project(self, quota_project_id):
        return self.__class__(
            self._source_credentials,
            self._credential_access_boundary,
            quota_project_id=quota_project_id,
        )
