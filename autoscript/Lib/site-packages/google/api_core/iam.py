# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Non-API-specific IAM policy definitions

For allowed roles / permissions, see:
https://cloud.google.com/iam/docs/understanding-roles

Example usage:

.. code-block:: python

   # ``get_iam_policy`` returns a :class:'~google.api_core.iam.Policy`.
   policy = resource.get_iam_policy(requested_policy_version=3)

   phred = "user:phred@example.com"
   admin_group = "group:admins@groups.example.com"
   account = "serviceAccount:account-1234@accounts.example.com"

   policy.version = 3
   policy.bindings = [
       {
           "role": "roles/owner",
           "members": {phred, admin_group, account}
       },
       {
           "role": "roles/editor",
           "members": {"allAuthenticatedUsers"}
       },
       {
           "role": "roles/viewer",
           "members": {"allUsers"}
           "condition": {
               "title": "request_time",
               "description": "Requests made before 2021-01-01T00:00:00Z",
               "expression": "request.time < timestamp(\"2021-01-01T00:00:00Z\")"
           }
       }
   ]

   resource.set_iam_policy(policy)
"""

import collections
import collections.abc
import operator
import warnings

# Generic IAM roles

OWNER_ROLE = "roles/owner"
"""Generic role implying all rights to an object."""

EDITOR_ROLE = "roles/editor"
"""Generic role implying rights to modify an object."""

VIEWER_ROLE = "roles/viewer"
"""Generic role implying rights to access an object."""

_ASSIGNMENT_DEPRECATED_MSG = """\
Assigning to '{}' is deprecated. Use the `policy.bindings` property to modify bindings instead."""

_DICT_ACCESS_MSG = """\
Dict access is not supported on policies with version > 1 or with conditional bindings."""


class InvalidOperationException(Exception):
    """Raised when trying to use Policy class as a dict."""

    pass


class Policy(collections.abc.MutableMapping):
    """IAM Policy

    Args:
        etag (Optional[str]): ETag used to identify a unique of the policy
        version (Optional[int]): The syntax schema version of the policy.

    Note:
        Using conditions in bindings requires the policy's version to be set
        to `3` or greater, depending on the versions that are currently supported.

        Accessing the policy using dict operations will raise InvalidOperationException
        when the policy's version is set to 3.

        Use the policy.bindings getter/setter to retrieve and modify the policy's bindings.

    See:
        IAM Policy https://cloud.google.com/iam/reference/rest/v1/Policy
        Policy versions https://cloud.google.com/iam/docs/policies#versions
        Conditions overview https://cloud.google.com/iam/docs/conditions-overview.
    """

    _OWNER_ROLES = (OWNER_ROLE,)
    """Roles mapped onto our ``owners`` attribute."""

    _EDITOR_ROLES = (EDITOR_ROLE,)
    """Roles mapped onto our ``editors`` attribute."""

    _VIEWER_ROLES = (VIEWER_ROLE,)
    """Roles mapped onto our ``viewers`` attribute."""

    def __init__(self, etag=None, version=None):
        self.etag = etag
        self.version = version
        self._bindings = []

    def __iter__(self):
        self.__check_version__()
        # Exclude bindings with no members
        return (binding["role"] for binding in self._bindings if binding["members"])

    def __len__(self):
        self.__check_version__()
        # Exclude bindings with no members
        return len(list(self.__iter__()))

    def __getitem__(self, key):
        self.__check_version__()
        for b in self._bindings:
            if b["role"] == key:
                return b["members"]
        # If the binding does not yet exist, create one
        # NOTE: This will create bindings with no members
        # which are ignored by __iter__ and __len__
        new_binding = {"role": key, "members": set()}
        self._bindings.append(new_binding)
        return new_binding["members"]

    def __setitem__(self, key, value):
        self.__check_version__()
        value = set(value)
        for binding in self._bindings:
            if binding["role"] == key:
                binding["members"] = value
                return
        self._bindings.append({"role": key, "members": value})

    def __delitem__(self, key):
        self.__check_version__()
        for b in self._bindings:
            if b["role"] == key:
                self._bindings.remove(b)
                return
        raise KeyError(key)

    def __check_version__(self):
        """Raise InvalidOperationException if version is greater than 1 or policy contains conditions."""
        raise_version = self.version is not None and self.version > 1

        if raise_version or self._contains_conditions():
            raise InvalidOperationException(_DICT_ACCESS_MSG)

    def _contains_conditions(self):
        for b in self._bindings:
            if b.get("condition") is not None:
                return True
        return False

    @property
    def bindings(self):
        """The policy's list of bindings.

        A binding is specified by a dictionary with keys:

        * role (str): Role that is assigned to `members`.

        * members (:obj:`set` of str): Specifies the identities associated to this binding.

        * condition (:obj:`dict` of str:str): Specifies a condition under which this binding will apply.

          * title (str): Title for the condition.

          * description (:obj:str, optional): Description of the condition.

          * expression: A CEL expression.

        Type:
           :obj:`list` of :obj:`dict`

        See:
           Policy versions https://cloud.google.com/iam/docs/policies#versions
           Conditions overview https://cloud.google.com/iam/docs/conditions-overview.

        Example:

        .. code-block:: python

           USER = "user:phred@example.com"
           ADMIN_GROUP = "group:admins@groups.example.com"
           SERVICE_ACCOUNT = "serviceAccount:account-1234@accounts.example.com"
           CONDITION = {
               "title": "request_time",
               "description": "Requests made before 2021-01-01T00:00:00Z", # Optional
               "expression": "request.time < timestamp(\"2021-01-01T00:00:00Z\")"
           }

           # Set policy's version to 3 before setting bindings containing conditions.
           policy.version = 3

           policy.bindings = [
               {
                   "role": "roles/viewer",
                   "members": {USER, ADMIN_GROUP, SERVICE_ACCOUNT},
                   "condition": CONDITION
               },
               ...
           ]
        """
        return self._bindings

    @bindings.setter
    def bindings(self, bindings):
        self._bindings = bindings

    @property
    def owners(self):
        """Legacy access to owner role.

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        DEPRECATED:  use `policy.bindings` to access bindings instead.
        """
        result = set()
        for role in self._OWNER_ROLES:
            for member in self.get(role, ()):
                result.add(member)
        return frozenset(result)

    @owners.setter
    def owners(self, value):
        """Update owners.

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        DEPRECATED:  use `policy.bindings` to access bindings instead.
        """
        warnings.warn(
            _ASSIGNMENT_DEPRECATED_MSG.format("owners", OWNER_ROLE), DeprecationWarning
        )
        self[OWNER_ROLE] = value

    @property
    def editors(self):
        """Legacy access to editor role.

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        DEPRECATED:  use `policy.bindings` to access bindings instead.
        """
        result = set()
        for role in self._EDITOR_ROLES:
            for member in self.get(role, ()):
                result.add(member)
        return frozenset(result)

    @editors.setter
    def editors(self, value):
        """Update editors.

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        DEPRECATED:  use `policy.bindings` to modify bindings instead.
        """
        warnings.warn(
            _ASSIGNMENT_DEPRECATED_MSG.format("editors", EDITOR_ROLE),
            DeprecationWarning,
        )
        self[EDITOR_ROLE] = value

    @property
    def viewers(self):
        """Legacy access to viewer role.

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        DEPRECATED:  use `policy.bindings` to modify bindings instead.
        """
        result = set()
        for role in self._VIEWER_ROLES:
            for member in self.get(role, ()):
                result.add(member)
        return frozenset(result)

    @viewers.setter
    def viewers(self, value):
        """Update viewers.

        Raise InvalidOperationException if version is greater than 1 or policy contains conditions.

        DEPRECATED:  use `policy.bindings` to modify bindings instead.
        """
        warnings.warn(
            _ASSIGNMENT_DEPRECATED_MSG.format("viewers", VIEWER_ROLE),
            DeprecationWarning,
        )
        self[VIEWER_ROLE] = value

    @staticmethod
    def user(email):
        """Factory method for a user member.

        Args:
            email (str): E-mail for this particular user.

        Returns:
            str: A member string corresponding to the given user.
        """
        return "user:%s" % (email,)

    @staticmethod
    def service_account(email):
        """Factory method for a service account member.

        Args:
            email (str): E-mail for this particular service account.

        Returns:
            str: A member string corresponding to the given service account.

        """
        return "serviceAccount:%s" % (email,)

    @staticmethod
    def group(email):
        """Factory method for a group member.

        Args:
            email (str): An id or e-mail for this particular group.

        Returns:
            str: A member string corresponding to the given group.
        """
        return "group:%s" % (email,)

    @staticmethod
    def domain(domain):
        """Factory method for a domain member.

        Args:
            domain (str): The domain for this member.

        Returns:
            str: A member string corresponding to the given domain.
        """
        return "domain:%s" % (domain,)

    @staticmethod
    def all_users():
        """Factory method for a member representing all users.

        Returns:
            str: A member string representing all users.
        """
        return "allUsers"

    @staticmethod
    def authenticated_users():
        """Factory method for a member representing all authenticated users.

        Returns:
            str: A member string representing all authenticated users.
        """
        return "allAuthenticatedUsers"

    @classmethod
    def from_api_repr(cls, resource):
        """Factory: create a policy from a JSON resource.

        Args:
            resource (dict): policy resource returned by ``getIamPolicy`` API.

        Returns:
            :class:`Policy`: the parsed policy
        """
        version = resource.get("version")
        etag = resource.get("etag")
        policy = cls(etag, version)
        policy.bindings = resource.get("bindings", [])

        for binding in policy.bindings:
            binding["members"] = set(binding.get("members", ()))

        return policy

    def to_api_repr(self):
        """Render a JSON policy resource.

        Returns:
            dict: a resource to be passed to the ``setIamPolicy`` API.
        """
        resource = {}

        if self.etag is not None:
            resource["etag"] = self.etag

        if self.version is not None:
            resource["version"] = self.version

        if self._bindings and len(self._bindings) > 0:
            bindings = []
            for binding in self._bindings:
                members = binding.get("members")
                if members:
                    new_binding = {"role": binding["role"], "members": sorted(members)}
                    condition = binding.get("condition")
                    if condition:
                        new_binding["condition"] = condition
                    bindings.append(new_binding)

            if bindings:
                # Sort bindings by role
                key = operator.itemgetter("role")
                resource["bindings"] = sorted(bindings, key=key)

        return resource
