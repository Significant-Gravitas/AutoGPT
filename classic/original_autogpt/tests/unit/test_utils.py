import json
import os
from pathlib import Path
from unittest.mock import patch

import autogpt.app.utils
import pytest
import requests
from autogpt.app.utils import (
    get_bulletin_from_web,
    get_current_git_branch,
    get_latest_bulletin,
    set_env_config_value,
)
from git import InvalidGitRepositoryError
from tests.utils import skip_in_ci

from forge.json.parsing import extract_dict_from_json


@pytest.fixture
def valid_json_response() -> dict:
    return {
        "thoughts": {
            "observations": "Retrieved Tesla's revenue data successfully.",
            "reasoning": "I will use the 'task_complete' command because it allows me "
            "to shut down and signal that my task is complete.",
            "plan": ["Use task_complete to shut down"],
            "self_criticism": "I need to ensure that I have completed all "
            "necessary tasks before shutting down.",
        },
        "command": {
            "name": "task_complete",
            "args": {"reason": "Task complete: retrieved Tesla's revenue in 2022."},
        },
    }


@pytest.fixture
def invalid_json_response() -> dict:
    return {
        "thoughts": {
            "observations": "Retrieved Tesla's revenue data.",
            "reasoning": "I will use the 'task_complete' command because it allows me "
            "to shut down and signal that my task is complete.",
            "plan": ["Use task_complete to shut down"],
            "self_criticism": "I need to ensure that I have completed all "
            "necessary tasks before shutting down.",
        },
        "command": {"name": "", "args": {}},
    }


@patch("requests.get")
def test_get_bulletin_from_web_success(mock_get):
    expected_content = "Test bulletin from web"

    mock_get.return_value.status_code = 200
    mock_get.return_value.text = expected_content
    bulletin = get_bulletin_from_web()

    assert expected_content in bulletin
    mock_get.assert_called_with(
        "https://raw.githubusercontent.com/Significant-Gravitas/AutoGPT/master/classic/original_autogpt/BULLETIN.md"  # noqa: E501
    )


@patch("requests.get")
def test_get_bulletin_from_web_failure(mock_get):
    mock_get.return_value.status_code = 404
    bulletin = get_bulletin_from_web()

    assert bulletin == ""


@patch("requests.get")
def test_get_bulletin_from_web_exception(mock_get):
    mock_get.side_effect = requests.exceptions.RequestException()
    bulletin = get_bulletin_from_web()

    assert bulletin == ""


def test_get_latest_bulletin_no_file(tmp_path, monkeypatch):
    bulletin_path = tmp_path / "data" / "CURRENT_BULLETIN.md"
    monkeypatch.chdir(tmp_path)
    # Ensure file doesn't exist
    if bulletin_path.exists():
        bulletin_path.unlink()

    with patch("autogpt.app.utils.get_bulletin_from_web", return_value=""):
        bulletin, is_new = get_latest_bulletin()
        assert is_new


def test_get_latest_bulletin_with_file(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    bulletin_path = data_dir / "CURRENT_BULLETIN.md"
    monkeypatch.chdir(tmp_path)

    expected_content = "Test bulletin"
    bulletin_path.write_text(expected_content, encoding="utf-8")

    with patch("autogpt.app.utils.get_bulletin_from_web", return_value=""):
        bulletin, is_new = get_latest_bulletin()
        assert expected_content in bulletin
        assert is_new is False


def test_get_latest_bulletin_with_new_bulletin(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    bulletin_path = data_dir / "CURRENT_BULLETIN.md"
    monkeypatch.chdir(tmp_path)

    bulletin_path.write_text("Old bulletin", encoding="utf-8")

    expected_content = "New bulletin from web"
    with patch(
        "autogpt.app.utils.get_bulletin_from_web", return_value=expected_content
    ):
        bulletin, is_new = get_latest_bulletin()
        assert "::NEW BULLETIN::" in bulletin
        assert expected_content in bulletin
        assert is_new


def test_get_latest_bulletin_new_bulletin_same_as_old_bulletin(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    bulletin_path = data_dir / "CURRENT_BULLETIN.md"
    monkeypatch.chdir(tmp_path)

    expected_content = "Current bulletin"
    bulletin_path.write_text(expected_content, encoding="utf-8")

    with patch(
        "autogpt.app.utils.get_bulletin_from_web", return_value=expected_content
    ):
        bulletin, is_new = get_latest_bulletin()
        assert expected_content in bulletin
        assert is_new is False


@skip_in_ci
def test_get_current_git_branch():
    branch_name = get_current_git_branch()
    assert branch_name != ""


@patch("autogpt.app.utils.Repo")
def test_get_current_git_branch_success(mock_repo):
    mock_repo.return_value.active_branch.name = "test-branch"
    branch_name = get_current_git_branch()

    assert branch_name == "test-branch"


@patch("autogpt.app.utils.Repo")
def test_get_current_git_branch_failure(mock_repo):
    mock_repo.side_effect = InvalidGitRepositoryError()
    branch_name = get_current_git_branch()

    assert branch_name == ""


def test_extract_json_from_response(valid_json_response: dict):
    emulated_response_from_openai = json.dumps(valid_json_response)
    assert extract_dict_from_json(emulated_response_from_openai) == valid_json_response


def test_extract_json_from_response_wrapped_in_code_block(valid_json_response: dict):
    emulated_response_from_openai = "```" + json.dumps(valid_json_response) + "```"
    assert extract_dict_from_json(emulated_response_from_openai) == valid_json_response


def test_extract_json_from_response_wrapped_in_code_block_with_language(
    valid_json_response: dict,
):
    emulated_response_from_openai = "```json" + json.dumps(valid_json_response) + "```"
    assert extract_dict_from_json(emulated_response_from_openai) == valid_json_response


def test_extract_json_from_response_json_contained_in_string(valid_json_response: dict):
    emulated_response_from_openai = (
        "sentence1" + json.dumps(valid_json_response) + "sentence2"
    )
    assert extract_dict_from_json(emulated_response_from_openai) == valid_json_response


@pytest.fixture
def mock_env_file_path(tmp_path):
    return tmp_path / ".env"


env_file_initial_content = """
# This is a comment
EXISTING_KEY=EXISTING_VALUE

## This is also a comment
# DISABLED_KEY=DISABLED_VALUE

# Another comment
UNUSED_KEY=UNUSED_VALUE
"""


@pytest.fixture
def mock_env_file(mock_env_file_path: Path, monkeypatch: pytest.MonkeyPatch):
    mock_env_file_path.write_text(env_file_initial_content)
    monkeypatch.setattr(autogpt.app.utils, "ENV_FILE_PATH", mock_env_file_path)
    return mock_env_file_path


@pytest.fixture
def mock_environ(monkeypatch: pytest.MonkeyPatch):
    env = {}
    monkeypatch.setattr(os, "environ", env)
    return env


def test_set_env_config_value_updates_existing_key(
    mock_env_file: Path, mock_environ: dict
):
    # Before updating, ensure the original content is as expected
    with mock_env_file.open("r") as file:
        assert file.readlines() == env_file_initial_content.splitlines(True)

    set_env_config_value("EXISTING_KEY", "NEW_VALUE")
    with mock_env_file.open("r") as file:
        content = file.readlines()

    # Ensure only the relevant line is altered
    expected_content_lines = [
        "\n",
        "# This is a comment\n",
        "EXISTING_KEY=NEW_VALUE\n",  # existing key + new value
        "\n",
        "## This is also a comment\n",
        "# DISABLED_KEY=DISABLED_VALUE\n",
        "\n",
        "# Another comment\n",
        "UNUSED_KEY=UNUSED_VALUE\n",
    ]
    assert content == expected_content_lines
    assert mock_environ["EXISTING_KEY"] == "NEW_VALUE"


def test_set_env_config_value_uncomments_and_updates_disabled_key(
    mock_env_file: Path, mock_environ: dict
):
    # Before adding, ensure the original content is as expected
    with mock_env_file.open("r") as file:
        assert file.readlines() == env_file_initial_content.splitlines(True)

    set_env_config_value("DISABLED_KEY", "ENABLED_NEW_VALUE")
    with mock_env_file.open("r") as file:
        content = file.readlines()

    # Ensure only the relevant line is altered
    expected_content_lines = [
        "\n",
        "# This is a comment\n",
        "EXISTING_KEY=EXISTING_VALUE\n",
        "\n",
        "## This is also a comment\n",
        "DISABLED_KEY=ENABLED_NEW_VALUE\n",  # disabled -> enabled + new value
        "\n",
        "# Another comment\n",
        "UNUSED_KEY=UNUSED_VALUE\n",
    ]
    assert content == expected_content_lines
    assert mock_environ["DISABLED_KEY"] == "ENABLED_NEW_VALUE"


def test_set_env_config_value_adds_new_key(mock_env_file: Path, mock_environ: dict):
    # Before adding, ensure the original content is as expected
    with mock_env_file.open("r") as file:
        assert file.readlines() == env_file_initial_content.splitlines(True)

    set_env_config_value("NEW_KEY", "NEW_VALUE")
    with mock_env_file.open("r") as file:
        content = file.readlines()

    # Ensure the new key-value pair is added without altering the rest
    expected_content_lines = [
        "\n",
        "# This is a comment\n",
        "EXISTING_KEY=EXISTING_VALUE\n",
        "\n",
        "## This is also a comment\n",
        "# DISABLED_KEY=DISABLED_VALUE\n",
        "\n",
        "# Another comment\n",
        "UNUSED_KEY=UNUSED_VALUE\n",
        "NEW_KEY=NEW_VALUE\n",  # New key-value pair added at the end
    ]
    assert content == expected_content_lines
    assert mock_environ["NEW_KEY"] == "NEW_VALUE"
