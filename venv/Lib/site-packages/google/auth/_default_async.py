# Copyright 2020 Google Inc.
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

"""Application default credentials.

Implements application default credentials and project ID detection.
"""

import io
import json
import os

import six

from google.auth import _default
from google.auth import environment_vars
from google.auth import exceptions


def load_credentials_from_file(filename, scopes=None, quota_project_id=None):
    """Loads Google credentials from a file.

    The credentials file must be a service account key or stored authorized
    user credentials.

    Args:
        filename (str): The full path to the credentials file.
        scopes (Optional[Sequence[str]]): The list of scopes for the credentials. If
            specified, the credentials will automatically be scoped if
            necessary
        quota_project_id (Optional[str]):  The project ID used for
                quota and billing.

    Returns:
        Tuple[google.auth.credentials.Credentials, Optional[str]]: Loaded
            credentials and the project ID. Authorized user credentials do not
            have the project ID information.

    Raises:
        google.auth.exceptions.DefaultCredentialsError: if the file is in the
            wrong format or is missing.
    """
    if not os.path.exists(filename):
        raise exceptions.DefaultCredentialsError(
            "File {} was not found.".format(filename)
        )

    with io.open(filename, "r") as file_obj:
        try:
            info = json.load(file_obj)
        except ValueError as caught_exc:
            new_exc = exceptions.DefaultCredentialsError(
                "File {} is not a valid json file.".format(filename), caught_exc
            )
            six.raise_from(new_exc, caught_exc)

    # The type key should indicate that the file is either a service account
    # credentials file or an authorized user credentials file.
    credential_type = info.get("type")

    if credential_type == _default._AUTHORIZED_USER_TYPE:
        from google.oauth2 import _credentials_async as credentials

        try:
            credentials = credentials.Credentials.from_authorized_user_info(
                info, scopes=scopes
            )
        except ValueError as caught_exc:
            msg = "Failed to load authorized user credentials from {}".format(filename)
            new_exc = exceptions.DefaultCredentialsError(msg, caught_exc)
            six.raise_from(new_exc, caught_exc)
        if quota_project_id:
            credentials = credentials.with_quota_project(quota_project_id)
        if not credentials.quota_project_id:
            _default._warn_about_problematic_credentials(credentials)
        return credentials, None

    elif credential_type == _default._SERVICE_ACCOUNT_TYPE:
        from google.oauth2 import _service_account_async as service_account

        try:
            credentials = service_account.Credentials.from_service_account_info(
                info, scopes=scopes
            ).with_quota_project(quota_project_id)
        except ValueError as caught_exc:
            msg = "Failed to load service account credentials from {}".format(filename)
            new_exc = exceptions.DefaultCredentialsError(msg, caught_exc)
            six.raise_from(new_exc, caught_exc)
        return credentials, info.get("project_id")

    else:
        raise exceptions.DefaultCredentialsError(
            "The file {file} does not have a valid type. "
            "Type is {type}, expected one of {valid_types}.".format(
                file=filename, type=credential_type, valid_types=_default._VALID_TYPES
            )
        )


def _get_gcloud_sdk_credentials(quota_project_id=None):
    """Gets the credentials and project ID from the Cloud SDK."""
    from google.auth import _cloud_sdk

    # Check if application default credentials exist.
    credentials_filename = _cloud_sdk.get_application_default_credentials_path()

    if not os.path.isfile(credentials_filename):
        return None, None

    credentials, project_id = load_credentials_from_file(
        credentials_filename, quota_project_id=quota_project_id
    )

    if not project_id:
        project_id = _cloud_sdk.get_project_id()

    return credentials, project_id


def _get_explicit_environ_credentials(quota_project_id=None):
    """Gets credentials from the GOOGLE_APPLICATION_CREDENTIALS environment
    variable."""
    from google.auth import _cloud_sdk

    cloud_sdk_adc_path = _cloud_sdk.get_application_default_credentials_path()
    explicit_file = os.environ.get(environment_vars.CREDENTIALS)

    if explicit_file is not None and explicit_file == cloud_sdk_adc_path:
        # Cloud sdk flow calls gcloud to fetch project id, so if the explicit
        # file path is cloud sdk credentials path, then we should fall back
        # to cloud sdk flow, otherwise project id cannot be obtained.
        return _get_gcloud_sdk_credentials(quota_project_id=quota_project_id)

    if explicit_file is not None:
        credentials, project_id = load_credentials_from_file(
            os.environ[environment_vars.CREDENTIALS], quota_project_id=quota_project_id
        )

        return credentials, project_id

    else:
        return None, None


def _get_gae_credentials():
    """Gets Google App Engine App Identity credentials and project ID."""
    # While this library is normally bundled with app_engine, there are
    # some cases where it's not available, so we tolerate ImportError.

    return _default._get_gae_credentials()


def _get_gce_credentials(request=None):
    """Gets credentials and project ID from the GCE Metadata Service."""
    # Ping requires a transport, but we want application default credentials
    # to require no arguments. So, we'll use the _http_client transport which
    # uses http.client. This is only acceptable because the metadata server
    # doesn't do SSL and never requires proxies.

    # While this library is normally bundled with compute_engine, there are
    # some cases where it's not available, so we tolerate ImportError.

    return _default._get_gce_credentials(request)


def default_async(scopes=None, request=None, quota_project_id=None):
    """Gets the default credentials for the current environment.

    `Application Default Credentials`_ provides an easy way to obtain
    credentials to call Google APIs for server-to-server or local applications.
    This function acquires credentials from the environment in the following
    order:

    1. If the environment variable ``GOOGLE_APPLICATION_CREDENTIALS`` is set
       to the path of a valid service account JSON private key file, then it is
       loaded and returned. The project ID returned is the project ID defined
       in the service account file if available (some older files do not
       contain project ID information).
    2. If the `Google Cloud SDK`_ is installed and has application default
       credentials set they are loaded and returned.

       To enable application default credentials with the Cloud SDK run::

            gcloud auth application-default login

       If the Cloud SDK has an active project, the project ID is returned. The
       active project can be set using::

            gcloud config set project

    3. If the application is running in the `App Engine standard environment`_
       (first generation) then the credentials and project ID from the
       `App Identity Service`_ are used.
    4. If the application is running in `Compute Engine`_ or `Cloud Run`_ or
       the `App Engine flexible environment`_ or the `App Engine standard
       environment`_ (second generation) then the credentials and project ID
       are obtained from the `Metadata Service`_.
    5. If no credentials are found,
       :class:`~google.auth.exceptions.DefaultCredentialsError` will be raised.

    .. _Application Default Credentials: https://developers.google.com\
            /identity/protocols/application-default-credentials
    .. _Google Cloud SDK: https://cloud.google.com/sdk
    .. _App Engine standard environment: https://cloud.google.com/appengine
    .. _App Identity Service: https://cloud.google.com/appengine/docs/python\
            /appidentity/
    .. _Compute Engine: https://cloud.google.com/compute
    .. _App Engine flexible environment: https://cloud.google.com\
            /appengine/flexible
    .. _Metadata Service: https://cloud.google.com/compute/docs\
            /storing-retrieving-metadata
    .. _Cloud Run: https://cloud.google.com/run

    Example::

        import google.auth

        credentials, project_id = google.auth.default()

    Args:
        scopes (Sequence[str]): The list of scopes for the credentials. If
            specified, the credentials will automatically be scoped if
            necessary.
        request (google.auth.transport.Request): An object used to make
            HTTP requests. This is used to detect whether the application
            is running on Compute Engine. If not specified, then it will
            use the standard library http client to make requests.
        quota_project_id (Optional[str]):  The project ID used for
            quota and billing.
    Returns:
        Tuple[~google.auth.credentials.Credentials, Optional[str]]:
            the current environment's credentials and project ID. Project ID
            may be None, which indicates that the Project ID could not be
            ascertained from the environment.

    Raises:
        ~google.auth.exceptions.DefaultCredentialsError:
            If no credentials were found, or if the credentials found were
            invalid.
    """
    from google.auth._credentials_async import with_scopes_if_required
    from google.auth.credentials import CredentialsWithQuotaProject

    explicit_project_id = os.environ.get(
        environment_vars.PROJECT, os.environ.get(environment_vars.LEGACY_PROJECT)
    )

    checkers = (
        lambda: _get_explicit_environ_credentials(quota_project_id=quota_project_id),
        lambda: _get_gcloud_sdk_credentials(quota_project_id=quota_project_id),
        _get_gae_credentials,
        lambda: _get_gce_credentials(request),
    )

    for checker in checkers:
        credentials, project_id = checker()
        if credentials is not None:
            credentials = with_scopes_if_required(credentials, scopes)
            if quota_project_id and isinstance(
                credentials, CredentialsWithQuotaProject
            ):
                credentials = credentials.with_quota_project(quota_project_id)
            effective_project_id = explicit_project_id or project_id
            if not effective_project_id:
                _default._LOGGER.warning(
                    "No project ID could be determined. Consider running "
                    "`gcloud config set project` or setting the %s "
                    "environment variable",
                    environment_vars.PROJECT,
                )
            return credentials, effective_project_id

    raise exceptions.DefaultCredentialsError(_default._CLOUD_SDK_MISSING_CREDENTIALS)
