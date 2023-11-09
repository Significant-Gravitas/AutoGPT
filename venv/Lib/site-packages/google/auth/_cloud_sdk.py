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

"""Helpers for reading the Google Cloud SDK's configuration."""

import os
import subprocess

import six

from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions


# The ~/.config subdirectory containing gcloud credentials.
_CONFIG_DIRECTORY = "gcloud"
# Windows systems store config at %APPDATA%\gcloud
_WINDOWS_CONFIG_ROOT_ENV_VAR = "APPDATA"
# The name of the file in the Cloud SDK config that contains default
# credentials.
_CREDENTIALS_FILENAME = "application_default_credentials.json"
# The name of the Cloud SDK shell script
_CLOUD_SDK_POSIX_COMMAND = "gcloud"
_CLOUD_SDK_WINDOWS_COMMAND = "gcloud.cmd"
# The command to get the Cloud SDK configuration
_CLOUD_SDK_CONFIG_GET_PROJECT_COMMAND = ("config", "get", "project")
# The command to get google user access token
_CLOUD_SDK_USER_ACCESS_TOKEN_COMMAND = ("auth", "print-access-token")
# Cloud SDK's application-default client ID
CLOUD_SDK_CLIENT_ID = (
    "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
)


def get_config_path():
    """Returns the absolute path the the Cloud SDK's configuration directory.

    Returns:
        str: The Cloud SDK config path.
    """
    # If the path is explicitly set, return that.
    try:
        return os.environ[environment_vars.CLOUD_SDK_CONFIG_DIR]
    except KeyError:
        pass

    # Non-windows systems store this at ~/.config/gcloud
    if os.name != "nt":
        return os.path.join(os.path.expanduser("~"), ".config", _CONFIG_DIRECTORY)
    # Windows systems store config at %APPDATA%\gcloud
    else:
        try:
            return os.path.join(
                os.environ[_WINDOWS_CONFIG_ROOT_ENV_VAR], _CONFIG_DIRECTORY
            )
        except KeyError:
            # This should never happen unless someone is really
            # messing with things, but we'll cover the case anyway.
            drive = os.environ.get("SystemDrive", "C:")
            return os.path.join(drive, "\\", _CONFIG_DIRECTORY)


def get_application_default_credentials_path():
    """Gets the path to the application default credentials file.

    The path may or may not exist.

    Returns:
        str: The full path to application default credentials.
    """
    config_path = get_config_path()
    return os.path.join(config_path, _CREDENTIALS_FILENAME)


def _run_subprocess_ignore_stderr(command):
    """ Return subprocess.check_output with the given command and ignores stderr."""
    with open(os.devnull, "w") as devnull:
        output = subprocess.check_output(command, stderr=devnull)
    return output


def get_project_id():
    """Gets the project ID from the Cloud SDK.

    Returns:
        Optional[str]: The project ID.
    """
    if os.name == "nt":
        command = _CLOUD_SDK_WINDOWS_COMMAND
    else:
        command = _CLOUD_SDK_POSIX_COMMAND

    try:
        # Ignore the stderr coming from gcloud, so it won't be mixed into the output.
        # https://github.com/googleapis/google-auth-library-python/issues/673
        project = _run_subprocess_ignore_stderr(
            (command,) + _CLOUD_SDK_CONFIG_GET_PROJECT_COMMAND
        )

        # Turn bytes into a string and remove "\n"
        project = _helpers.from_bytes(project).strip()
        return project if project else None
    except (subprocess.CalledProcessError, OSError, IOError):
        return None


def get_auth_access_token(account=None):
    """Load user access token with the ``gcloud auth print-access-token`` command.

    Args:
        account (Optional[str]): Account to get the access token for. If not
            specified, the current active account will be used.

    Returns:
        str: The user access token.

    Raises:
        google.auth.exceptions.UserAccessTokenError: if failed to get access
            token from gcloud.
    """
    if os.name == "nt":
        command = _CLOUD_SDK_WINDOWS_COMMAND
    else:
        command = _CLOUD_SDK_POSIX_COMMAND

    try:
        if account:
            command = (
                (command,)
                + _CLOUD_SDK_USER_ACCESS_TOKEN_COMMAND
                + ("--account=" + account,)
            )
        else:
            command = (command,) + _CLOUD_SDK_USER_ACCESS_TOKEN_COMMAND

        access_token = subprocess.check_output(command, stderr=subprocess.STDOUT)
        # remove the trailing "\n"
        return access_token.decode("utf-8").strip()
    except (subprocess.CalledProcessError, OSError, IOError) as caught_exc:
        new_exc = exceptions.UserAccessTokenError(
            "Failed to obtain access token", caught_exc
        )
        six.raise_from(new_exc, caught_exc)
