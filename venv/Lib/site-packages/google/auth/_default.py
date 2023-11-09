# Copyright 2015 Google Inc.
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
import logging
import os
import warnings

import six

from google.auth import environment_vars
from google.auth import exceptions
import google.auth.transport._http_client

_LOGGER = logging.getLogger(__name__)

# Valid types accepted for file-based credentials.
_AUTHORIZED_USER_TYPE = "authorized_user"
_SERVICE_ACCOUNT_TYPE = "service_account"
_EXTERNAL_ACCOUNT_TYPE = "external_account"
_EXTERNAL_ACCOUNT_AUTHORIZED_USER_TYPE = "external_account_authorized_user"
_IMPERSONATED_SERVICE_ACCOUNT_TYPE = "impersonated_service_account"
_GDCH_SERVICE_ACCOUNT_TYPE = "gdch_service_account"
_VALID_TYPES = (
    _AUTHORIZED_USER_TYPE,
    _SERVICE_ACCOUNT_TYPE,
    _EXTERNAL_ACCOUNT_TYPE,
    _EXTERNAL_ACCOUNT_AUTHORIZED_USER_TYPE,
    _IMPERSONATED_SERVICE_ACCOUNT_TYPE,
    _GDCH_SERVICE_ACCOUNT_TYPE,
)

# Help message when no credentials can be found.
_CLOUD_SDK_MISSING_CREDENTIALS = """\
Your default credentials were not found. To set up Application Default Credentials, \
see https://cloud.google.com/docs/authentication/external/set-up-adc for more information.\
"""

# Warning when using Cloud SDK user credentials
_CLOUD_SDK_CREDENTIALS_WARNING = """\
Your application has authenticated using end user credentials from Google \
Cloud SDK without a quota project. You might receive a "quota exceeded" \
or "API not enabled" error. See the following page for troubleshooting: \
https://cloud.google.com/docs/authentication/adc-troubleshooting/user-creds. \
"""

# The subject token type used for AWS external_account credentials.
_AWS_SUBJECT_TOKEN_TYPE = "urn:ietf:params:aws:token-type:aws4_request"


def _warn_about_problematic_credentials(credentials):
    """Determines if the credentials are problematic.

    Credentials from the Cloud SDK that are associated with Cloud SDK's project
    are problematic because they may not have APIs enabled and have limited
    quota. If this is the case, warn about it.
    """
    from google.auth import _cloud_sdk

    if credentials.client_id == _cloud_sdk.CLOUD_SDK_CLIENT_ID:
        warnings.warn(_CLOUD_SDK_CREDENTIALS_WARNING)


def load_credentials_from_file(
    filename, scopes=None, default_scopes=None, quota_project_id=None, request=None
):
    """Loads Google credentials from a file.

    The credentials file must be a service account key, stored authorized
    user credentials, external account credentials, or impersonated service
    account credentials.

    Args:
        filename (str): The full path to the credentials file.
        scopes (Optional[Sequence[str]]): The list of scopes for the credentials. If
            specified, the credentials will automatically be scoped if
            necessary
        default_scopes (Optional[Sequence[str]]): Default scopes passed by a
            Google client library. Use 'scopes' for user-defined scopes.
        quota_project_id (Optional[str]):  The project ID used for
            quota and billing.
        request (Optional[google.auth.transport.Request]): An object used to make
            HTTP requests. This is used to determine the associated project ID
            for a workload identity pool resource (external account credentials).
            If not specified, then it will use a
            google.auth.transport.requests.Request client to make requests.

    Returns:
        Tuple[google.auth.credentials.Credentials, Optional[str]]: Loaded
            credentials and the project ID. Authorized user credentials do not
            have the project ID information. External account credentials project
            IDs may not always be determined.

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
    return _load_credentials_from_info(
        filename, info, scopes, default_scopes, quota_project_id, request
    )


def _load_credentials_from_info(
    filename, info, scopes, default_scopes, quota_project_id, request
):
    from google.auth.credentials import CredentialsWithQuotaProject

    credential_type = info.get("type")

    if credential_type == _AUTHORIZED_USER_TYPE:
        credentials, project_id = _get_authorized_user_credentials(
            filename, info, scopes
        )

    elif credential_type == _SERVICE_ACCOUNT_TYPE:
        credentials, project_id = _get_service_account_credentials(
            filename, info, scopes, default_scopes
        )

    elif credential_type == _EXTERNAL_ACCOUNT_TYPE:
        credentials, project_id = _get_external_account_credentials(
            info,
            filename,
            scopes=scopes,
            default_scopes=default_scopes,
            request=request,
        )

    elif credential_type == _EXTERNAL_ACCOUNT_AUTHORIZED_USER_TYPE:
        credentials, project_id = _get_external_account_authorized_user_credentials(
            filename, info, request
        )

    elif credential_type == _IMPERSONATED_SERVICE_ACCOUNT_TYPE:
        credentials, project_id = _get_impersonated_service_account_credentials(
            filename, info, scopes
        )
    elif credential_type == _GDCH_SERVICE_ACCOUNT_TYPE:
        credentials, project_id = _get_gdch_service_account_credentials(filename, info)
    else:
        raise exceptions.DefaultCredentialsError(
            "The file {file} does not have a valid type. "
            "Type is {type}, expected one of {valid_types}.".format(
                file=filename, type=credential_type, valid_types=_VALID_TYPES
            )
        )
    if isinstance(credentials, CredentialsWithQuotaProject):
        credentials = _apply_quota_project_id(credentials, quota_project_id)
    return credentials, project_id


def _get_gcloud_sdk_credentials(quota_project_id=None):
    """Gets the credentials and project ID from the Cloud SDK."""
    from google.auth import _cloud_sdk

    _LOGGER.debug("Checking Cloud SDK credentials as part of auth process...")

    # Check if application default credentials exist.
    credentials_filename = _cloud_sdk.get_application_default_credentials_path()

    if not os.path.isfile(credentials_filename):
        _LOGGER.debug("Cloud SDK credentials not found on disk; not using them")
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

    _LOGGER.debug(
        "Checking %s for explicit credentials as part of auth process...", explicit_file
    )

    if explicit_file is not None and explicit_file == cloud_sdk_adc_path:
        # Cloud sdk flow calls gcloud to fetch project id, so if the explicit
        # file path is cloud sdk credentials path, then we should fall back
        # to cloud sdk flow, otherwise project id cannot be obtained.
        _LOGGER.debug(
            "Explicit credentials path %s is the same as Cloud SDK credentials path, fall back to Cloud SDK credentials flow...",
            explicit_file,
        )
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
    # If not GAE gen1, prefer the metadata service even if the GAE APIs are
    # available as per https://google.aip.dev/auth/4115.
    if os.environ.get(environment_vars.LEGACY_APPENGINE_RUNTIME) != "python27":
        return None, None

    # While this library is normally bundled with app_engine, there are
    # some cases where it's not available, so we tolerate ImportError.
    try:
        _LOGGER.debug("Checking for App Engine runtime as part of auth process...")
        import google.auth.app_engine as app_engine
    except ImportError:
        _LOGGER.warning("Import of App Engine auth library failed.")
        return None, None

    try:
        credentials = app_engine.Credentials()
        project_id = app_engine.get_project_id()
        return credentials, project_id
    except EnvironmentError:
        _LOGGER.debug(
            "No App Engine library was found so cannot authentication via App Engine Identity Credentials."
        )
        return None, None


def _get_gce_credentials(request=None, quota_project_id=None):
    """Gets credentials and project ID from the GCE Metadata Service."""
    # Ping requires a transport, but we want application default credentials
    # to require no arguments. So, we'll use the _http_client transport which
    # uses http.client. This is only acceptable because the metadata server
    # doesn't do SSL and never requires proxies.

    # While this library is normally bundled with compute_engine, there are
    # some cases where it's not available, so we tolerate ImportError.
    try:
        from google.auth import compute_engine
        from google.auth.compute_engine import _metadata
    except ImportError:
        _LOGGER.warning("Import of Compute Engine auth library failed.")
        return None, None

    if request is None:
        request = google.auth.transport._http_client.Request()

    if _metadata.ping(request=request):
        # Get the project ID.
        try:
            project_id = _metadata.get_project_id(request=request)
        except exceptions.TransportError:
            project_id = None

        cred = compute_engine.Credentials()
        cred = _apply_quota_project_id(cred, quota_project_id)

        return cred, project_id
    else:
        _LOGGER.warning(
            "Authentication failed using Compute Engine authentication due to unavailable metadata server."
        )
        return None, None


def _get_external_account_credentials(
    info, filename, scopes=None, default_scopes=None, request=None
):
    """Loads external account Credentials from the parsed external account info.

    The credentials information must correspond to a supported external account
    credentials.

    Args:
        info (Mapping[str, str]): The external account info in Google format.
        filename (str): The full path to the credentials file.
        scopes (Optional[Sequence[str]]): The list of scopes for the credentials. If
            specified, the credentials will automatically be scoped if
            necessary.
        default_scopes (Optional[Sequence[str]]): Default scopes passed by a
            Google client library. Use 'scopes' for user-defined scopes.
        request (Optional[google.auth.transport.Request]): An object used to make
            HTTP requests. This is used to determine the associated project ID
            for a workload identity pool resource (external account credentials).
            If not specified, then it will use a
            google.auth.transport.requests.Request client to make requests.

    Returns:
        Tuple[google.auth.credentials.Credentials, Optional[str]]: Loaded
            credentials and the project ID. External account credentials project
            IDs may not always be determined.

    Raises:
        google.auth.exceptions.DefaultCredentialsError: if the info dictionary
            is in the wrong format or is missing required information.
    """
    # There are currently 3 types of external_account credentials.
    if info.get("subject_token_type") == _AWS_SUBJECT_TOKEN_TYPE:
        # Check if configuration corresponds to an AWS credentials.
        from google.auth import aws

        credentials = aws.Credentials.from_info(
            info, scopes=scopes, default_scopes=default_scopes
        )
    elif (
        info.get("credential_source") is not None
        and info.get("credential_source").get("executable") is not None
    ):
        from google.auth import pluggable

        credentials = pluggable.Credentials.from_info(
            info, scopes=scopes, default_scopes=default_scopes
        )
    else:
        try:
            # Check if configuration corresponds to an Identity Pool credentials.
            from google.auth import identity_pool

            credentials = identity_pool.Credentials.from_info(
                info, scopes=scopes, default_scopes=default_scopes
            )
        except ValueError:
            # If the configuration is invalid or does not correspond to any
            # supported external_account credentials, raise an error.
            raise exceptions.DefaultCredentialsError(
                "Failed to load external account credentials from {}".format(filename)
            )
    if request is None:
        import google.auth.transport.requests

        request = google.auth.transport.requests.Request()

    return credentials, credentials.get_project_id(request=request)


def _get_external_account_authorized_user_credentials(
    filename, info, scopes=None, default_scopes=None, request=None
):
    try:
        from google.auth import external_account_authorized_user

        credentials = external_account_authorized_user.Credentials.from_info(info)
    except ValueError:
        raise exceptions.DefaultCredentialsError(
            "Failed to load external account authorized user credentials from {}".format(
                filename
            )
        )

    return credentials, None


def _get_authorized_user_credentials(filename, info, scopes=None):
    from google.oauth2 import credentials

    try:
        credentials = credentials.Credentials.from_authorized_user_info(
            info, scopes=scopes
        )
    except ValueError as caught_exc:
        msg = "Failed to load authorized user credentials from {}".format(filename)
        new_exc = exceptions.DefaultCredentialsError(msg, caught_exc)
        six.raise_from(new_exc, caught_exc)
    return credentials, None


def _get_service_account_credentials(filename, info, scopes=None, default_scopes=None):
    from google.oauth2 import service_account

    try:
        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=scopes, default_scopes=default_scopes
        )
    except ValueError as caught_exc:
        msg = "Failed to load service account credentials from {}".format(filename)
        new_exc = exceptions.DefaultCredentialsError(msg, caught_exc)
        six.raise_from(new_exc, caught_exc)
    return credentials, info.get("project_id")


def _get_impersonated_service_account_credentials(filename, info, scopes):
    from google.auth import impersonated_credentials

    try:
        source_credentials_info = info.get("source_credentials")
        source_credentials_type = source_credentials_info.get("type")
        if source_credentials_type == _AUTHORIZED_USER_TYPE:
            source_credentials, _ = _get_authorized_user_credentials(
                filename, source_credentials_info
            )
        elif source_credentials_type == _SERVICE_ACCOUNT_TYPE:
            source_credentials, _ = _get_service_account_credentials(
                filename, source_credentials_info
            )
        else:
            raise exceptions.InvalidType(
                "source credential of type {} is not supported.".format(
                    source_credentials_type
                )
            )
        impersonation_url = info.get("service_account_impersonation_url")
        start_index = impersonation_url.rfind("/")
        end_index = impersonation_url.find(":generateAccessToken")
        if start_index == -1 or end_index == -1 or start_index > end_index:
            raise exceptions.InvalidValue(
                "Cannot extract target principal from {}".format(impersonation_url)
            )
        target_principal = impersonation_url[start_index + 1 : end_index]
        delegates = info.get("delegates")
        quota_project_id = info.get("quota_project_id")
        credentials = impersonated_credentials.Credentials(
            source_credentials,
            target_principal,
            scopes,
            delegates,
            quota_project_id=quota_project_id,
        )
    except ValueError as caught_exc:
        msg = "Failed to load impersonated service account credentials from {}".format(
            filename
        )
        new_exc = exceptions.DefaultCredentialsError(msg, caught_exc)
        six.raise_from(new_exc, caught_exc)
    return credentials, None


def _get_gdch_service_account_credentials(filename, info):
    from google.oauth2 import gdch_credentials

    try:
        credentials = gdch_credentials.ServiceAccountCredentials.from_service_account_info(
            info
        )
    except ValueError as caught_exc:
        msg = "Failed to load GDCH service account credentials from {}".format(filename)
        new_exc = exceptions.DefaultCredentialsError(msg, caught_exc)
        six.raise_from(new_exc, caught_exc)
    return credentials, info.get("project")


def get_api_key_credentials(key):
    """Return credentials with the given API key."""
    from google.auth import api_key

    return api_key.Credentials(key)


def _apply_quota_project_id(credentials, quota_project_id):
    if quota_project_id:
        credentials = credentials.with_quota_project(quota_project_id)
    else:
        credentials = credentials.with_quota_project_from_environment()

    from google.oauth2 import credentials as authorized_user_credentials

    if isinstance(credentials, authorized_user_credentials.Credentials) and (
        not credentials.quota_project_id
    ):
        _warn_about_problematic_credentials(credentials)
    return credentials


def default(scopes=None, request=None, quota_project_id=None, default_scopes=None):
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

       If the environment variable is set to the path of a valid external
       account JSON configuration file (workload identity federation), then the
       configuration file is used to determine and retrieve the external
       credentials from the current environment (AWS, Azure, etc).
       These will then be exchanged for Google access tokens via the Google STS
       endpoint.
       The project ID returned in this case is the one corresponding to the
       underlying workload identity pool resource if determinable.

       If the environment variable is set to the path of a valid GDCH service
       account JSON file (`Google Distributed Cloud Hosted`_), then a GDCH
       credential will be returned. The project ID returned is the project
       specified in the JSON file.
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
    .. _Google Distributed Cloud Hosted: https://cloud.google.com/blog/topics\
            /hybrid-cloud/announcing-google-distributed-cloud-edge-and-hosted

    Example::

        import google.auth

        credentials, project_id = google.auth.default()

    Args:
        scopes (Sequence[str]): The list of scopes for the credentials. If
            specified, the credentials will automatically be scoped if
            necessary.
        request (Optional[google.auth.transport.Request]): An object used to make
            HTTP requests. This is used to either detect whether the application
            is running on Compute Engine or to determine the associated project
            ID for a workload identity pool resource (external account
            credentials). If not specified, then it will either use the standard
            library http client to make requests for Compute Engine credentials
            or a google.auth.transport.requests.Request client for external
            account credentials.
        quota_project_id (Optional[str]): The project ID used for
            quota and billing.
        default_scopes (Optional[Sequence[str]]): Default scopes passed by a
            Google client library. Use 'scopes' for user-defined scopes.
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
    from google.auth.credentials import with_scopes_if_required
    from google.auth.credentials import CredentialsWithQuotaProject

    explicit_project_id = os.environ.get(
        environment_vars.PROJECT, os.environ.get(environment_vars.LEGACY_PROJECT)
    )

    checkers = (
        # Avoid passing scopes here to prevent passing scopes to user credentials.
        # with_scopes_if_required() below will ensure scopes/default scopes are
        # safely set on the returned credentials since requires_scopes will
        # guard against setting scopes on user credentials.
        lambda: _get_explicit_environ_credentials(quota_project_id=quota_project_id),
        lambda: _get_gcloud_sdk_credentials(quota_project_id=quota_project_id),
        _get_gae_credentials,
        lambda: _get_gce_credentials(request, quota_project_id=quota_project_id),
    )

    for checker in checkers:
        credentials, project_id = checker()
        if credentials is not None:
            credentials = with_scopes_if_required(
                credentials, scopes, default_scopes=default_scopes
            )

            # For external account credentials, scopes are required to determine
            # the project ID. Try to get the project ID again if not yet
            # determined.
            if not project_id and callable(
                getattr(credentials, "get_project_id", None)
            ):
                if request is None:
                    import google.auth.transport.requests

                    request = google.auth.transport.requests.Request()
                project_id = credentials.get_project_id(request=request)

            if quota_project_id and isinstance(
                credentials, CredentialsWithQuotaProject
            ):
                credentials = credentials.with_quota_project(quota_project_id)

            effective_project_id = explicit_project_id or project_id
            if not effective_project_id:
                _LOGGER.warning(
                    "No project ID could be determined. Consider running "
                    "`gcloud config set project` or setting the %s "
                    "environment variable",
                    environment_vars.PROJECT,
                )
            return credentials, effective_project_id

    raise exceptions.DefaultCredentialsError(_CLOUD_SDK_MISSING_CREDENTIALS)
