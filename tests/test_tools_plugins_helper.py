from __future__ import annotations

import os
import subprocess
from unittest.mock import Mock, patch

import pytest

from AFAAS.core.tools.tool import Tool
from AFAAS.lib.sdk.add_api_key import ensure_api_key, install_and_import_package
from AFAAS.plugins.tools.langchain_google_places import (
    GooglePlacesAPIWrapper,
    GooglePlacesTool,
)


# Example revised test case
@pytest.fixture
def setup_env():
    # Setup for each test
    original_env = dict(os.environ)
    yield
    # Teardown after each test
    os.environ = original_env


@patch.dict(os.environ, {"POETRY_ACTIVE": "0"})
@patch("builtins.__import__")
@patch("AFAAS.lib.sdk.add_api_key.subprocess.run")
def test_successful_package_installation(mock_subprocess_run, mock_import, setup_env):
    mock_subprocess_run.return_value.returncode = 0
    mock_import.side_effect = ImportError  # Simplified side effect
    os.environ["POETRY_ACTIVE"] = "0"
    with pytest.raises(ImportError):
        install_and_import_package("nonexistent_package")

    mock_subprocess_run.assert_called_with(
        ["pip", "install", "nonexistent_package"], check=False
    )


@patch.dict(os.environ, {"POETRY_ACTIVE": "0"})
@patch("builtins.__import__")
@patch("subprocess.run")
def test_successful_package_installationv2(mock_subprocess_run, mock_import, setup_env):
    mock_subprocess_run.return_value.returncode = 0
    mock_import.side_effect = ImportError  # Simplified side effect
    os.environ["POETRY_ACTIVE"] = "0"
    with pytest.raises(ImportError):
        install_and_import_package("nonexistent_package")

    mock_subprocess_run.assert_called_with(
        ["pip", "install", "nonexistent_package"], check=False
    )


@patch("builtins.__import__")
@patch("subprocess.run")
def test_existing_package(mock_subprocess_run, mock_import):
    # Simulate successful import without ImportError
    mock_import.return_value = Mock()

    install_and_import_package("existing_package")

    mock_subprocess_run.assert_not_called()
    mock_import.assert_called_with("existing_package")


@patch("builtins.__import__")
@patch("subprocess.run")
def test_installation_failure_handling(mock_subprocess_run, mock_import, setup_env):
    mock_subprocess_run.return_value.returncode = 1  # Simulate installation failure
    mock_import.side_effect = ImportError("No module named 'failing_package'")

    with pytest.raises(SystemExit):
        install_and_import_package("failing_package")

    mock_subprocess_run.assert_called_with(
        ["pip", "install", "failing_package"], check=False
    )
    assert mock_import.call_count == 1


@patch.dict(os.environ, {"POETRY_ACTIVE": "1"})
@patch("builtins.__import__")
@patch("subprocess.run")
def test_successful_package_installation_poetry(mock_subprocess_run, mock_import):
    mock_subprocess_run.return_value.returncode = 0
    mock_import.side_effect = [ImportError, None]
    os.environ["POETRY_ACTIVE"] = "1"
    install_and_import_package("nonexistent_poetry_package")

    mock_subprocess_run.assert_called_with(
        ["poetry", "add", "nonexistent_poetry_package"], check=False
    )
    mock_import.assert_called_with("nonexistent_poetry_package")


@patch("builtins.__import__")
@patch("subprocess.run")
@patch("sys.exit")
def test_import_error_on_second_attempt(
    mock_sys_exit, mock_subprocess_run, mock_import
):
    mock_subprocess_run.return_value.returncode = 765
    mock_import.side_effect = [ImportError, ImportError]
    os.environ["POETRY_ACTIVE"] = "0"

    with pytest.raises(ImportError):
        install_and_import_package("package_import_error")

    mock_subprocess_run.assert_called_with(
        ["pip", "install", "package_import_error"], check=False
    )
    mock_import.assert_called_with("package_import_error")
    mock_sys_exit.assert_called_once()


@patch("builtins.__import__")
@patch("subprocess.run")
@patch("sys.exit")
def test_invalid_package_name(mock_sys_exit, mock_subprocess_run, mock_import):
    mock_subprocess_run.return_value.returncode = 1
    mock_import.side_effect = ImportError
    os.environ["POETRY_ACTIVE"] = "0"

    with pytest.raises(ImportError):
        install_and_import_package("invalid_package")

    mock_subprocess_run.assert_called_with(
        ["pip", "install", "invalid_package"], check=False
    )
    mock_import.assert_called_with("invalid_package")
    mock_sys_exit.assert_called_once()


@patch("builtins.__import__")
@patch("subprocess.run")
@patch("sys.exit")
def test_network_issues_during_installation(
    mock_sys_exit, mock_subprocess_run, mock_import
):
    # Simulate network failure with a non-zero return code
    mock_subprocess_run.return_value.returncode = 1  # Non-zero return code
    mock_import.side_effect = ImportError

    with pytest.raises(ImportError):
        install_and_import_package("package_with_network_issue")

    mock_subprocess_run.assert_called_with(
        ["pip", "install", "package_with_network_issue"], check=False
    )
    mock_import.assert_called_with("package_with_network_issue")
    mock_sys_exit.assert_called_once()


@patch("builtins.__import__")
@patch("subprocess.run")
@patch("sys.exit")
def test_permission_issues_during_installation(
    mock_sys_exit, mock_subprocess_run, mock_import
):
    # Simulate permission failure with a non-zero return code
    mock_subprocess_run.return_value.returncode = 1  # Non-zero return code
    mock_import.side_effect = ImportError

    with pytest.raises(ImportError):
        install_and_import_package("package_with_permission_issue")

    mock_subprocess_run.assert_called_with(
        ["pip", "install", "package_with_permission_issue"], check=False
    )
    mock_import.assert_called_with("package_with_permission_issue")
    mock_sys_exit.assert_called_once()


@patch.dict(os.environ, {"POETRY_ACTIVE": "1"})
@patch("builtins.__import__")
@patch("subprocess.run")
@patch("sys.exit")
def test_failed_package_installation_poetry(
    mock_sys_exit, mock_subprocess_run, mock_import
):
    mock_subprocess_run.return_value.returncode = 1  # Simulate installation failure
    mock_import.side_effect = ImportError
    os.environ["POETRY_ACTIVE"] = "1"

    with pytest.raises(ImportError):
        install_and_import_package("failing_poetry_package")

    mock_subprocess_run.assert_called_with(
        ["poetry", "add", "failing_poetry_package"], check=False
    )
    mock_import.assert_called_with("failing_poetry_package")
    mock_sys_exit.assert_called_once()
